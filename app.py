import os
import base64
import smtplib
from urllib.parse import urlparse

import numpy as np
import cv2
import gridfs
import requests
from bson import ObjectId
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pymongo import MongoClient
import face_recognition

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/checkmate")

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
ALERTS_EMAIL = os.environ.get("ALERTS_EMAIL")

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL")  # e.g. https://checkmate....appdomain.cloud
IMAGE_MAX_BYTES = int(os.environ.get("IMAGE_MAX_BYTES", 8_000_000))  # 8MB default

# Comma-separated domains allowed for image_url fetch (SSRF protection)
IMAGE_URL_ALLOWLIST = os.environ.get(
    "IMAGE_URL_ALLOWLIST",
    "api.leadconnectorhq.com,link.msgsndr.com,storage.googleapis.com"
)

# --- DATABASE SETUP ---
client = MongoClient(MONGO_URI)
db = client.get_default_database()
faces_collection = db.known_faces
fs = gridfs.GridFS(db)


def send_alert_email(name: str) -> None:
    if not all([SMTP_USER, SMTP_PASS, ALERTS_EMAIL]):
        return

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = ALERTS_EMAIL
    msg["Subject"] = f"ALERT: {name} Identified"
    msg.attach(MIMEText(f"The individual {name} was identified in a scan.", "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Email error: {e}")


def _is_allowed_image_url(u: str) -> bool:
    try:
        parsed = urlparse(u)
        if parsed.scheme not in ("https", "http"):
            return False

        host = (parsed.hostname or "").lower()
        allow = [d.strip().lower() for d in IMAGE_URL_ALLOWLIST.split(",") if d.strip()]

        # Exact match or subdomain match
        return any(host == d or host.endswith("." + d) for d in allow)
    except Exception:
        return False


def _bytes_from_image_data_or_url(data: dict) -> tuple[bytes, str]:
    """
    Returns (image_bytes, content_type).
    Supports either:
      - data['image_data'] as a data URL (data:image/jpeg;base64,...) or raw base64
      - data['image_url'] as a remote URL (fetched server-side, allowlisted)
    """
    image_data = data.get("image_data")
    image_url = data.get("image_url")

    if image_data:
        content_type = "application/octet-stream"
        if "," in image_data and image_data.strip().lower().startswith("data:"):
            header, encoded = image_data.split(",", 1)
            # header like: data:image/jpeg;base64
            try:
                content_type = header.split(":", 1)[1].split(";", 1)[0].strip() or content_type
            except Exception:
                pass
        else:
            encoded = image_data

        image_bytes = base64.b64decode(encoded)
        if not image_bytes:
            raise ValueError("Empty image_data after decoding.")
        if len(image_bytes) > IMAGE_MAX_BYTES:
            raise ValueError("image_data exceeds max size.")
        return image_bytes, content_type

    if image_url:
        if not _is_allowed_image_url(image_url):
            raise ValueError("image_url domain not allowed.")

        r = requests.get(image_url, timeout=12, stream=True)
        r.raise_for_status()
        content_type = (r.headers.get("Content-Type") or "application/octet-stream").split(";")[0].strip()

        # Read up to IMAGE_MAX_BYTES
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > IMAGE_MAX_BYTES:
                raise ValueError("image_url download exceeds max size.")
            chunks.append(chunk)

        image_bytes = b"".join(chunks)
        if not image_bytes:
            raise ValueError("Empty image_url response body.")
        return image_bytes, content_type

    raise ValueError("Missing image_data or image_url.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["GET", "POST"], strict_slashes=False)
def search():
    return render_template("search.html")


@app.route("/results", methods=["GET", "POST"], strict_slashes=False)
def results():
    return render_template("results.html")


# --- ROUTE TO SERVE IMAGES ---
@app.route("/image/<file_id>")
def serve_image(file_id):
    try:
        grid_out = fs.get(ObjectId(file_id))
        mimetype = getattr(grid_out, "content_type", None) or "application/octet-stream"
        return send_file(grid_out, mimetype=mimetype)
    except Exception:
        return "Image not found", 404


@app.route("/submit-and-check", methods=["POST"])
def submit_and_check():
    try:
        data = request.get_json(silent=True) or {}

        raw_name = (data.get("name") or "").strip()
        # Avoid poisoning DB with one shared "unknown" name.
        normalized_name = raw_name.lower().strip() if raw_name else f"subject_{ObjectId()}"

        # Input validation / bytes retrieval
        try:
            image_bytes, content_type = _bytes_from_image_data_or_url(data)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400

        # --- STORAGE IN GRIDFS ---
        file_id = fs.put(image_bytes, content_type=content_type)
        relative_image_url = f"/image/{file_id}"
        base = (PUBLIC_BASE_URL or request.host_url.rstrip("/"))
        image_url = f"{base}{relative_image_url}"

        # Decode for face recognition
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Invalid image bytes."}), 400

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        unknown_encodings = face_recognition.face_encodings(rgb_img)
        if not unknown_encodings:
            return jsonify({"status": "no_face_detected"}), 200

        known_faces_data = list(faces_collection.find())
        display_name = "Unknown"
        history = []
        overlaps = []

        # Build valid known encodings list
        known_encs = []
        known_names = []
        for f in known_faces_data:
            enc = f.get("encoding")
            nm = f.get("name")
            if nm and isinstance(enc, (list, tuple)) and len(enc) == 128:
                known_encs.append(np.array(enc, dtype=np.float64))
                known_names.append(nm)

        # --- FIX #1: Guard compare_faces when list is empty ---
        if known_encs:
            matches = face_recognition.compare_faces(
                known_encs,
                unknown_encodings[0],
                tolerance=0.6
            )

            if True in matches:
                display_name = known_names[matches.index(True)]
                history = list(
                    faces_collection.find(
                        {"name": display_name},
                        {"_id": 0, "encoding": 0, "submitter_email": 0}
                    )
                )

                user_start = data.get("start_date")
                user_end = data.get("end_date") or "9999-12-31"

                for record in history:
                    rec_start = record.get("start_date")
                    rec_end = record.get("end_date") or "9999-12-31"
                    if user_start and rec_start:
                        if user_start <= rec_end and user_end >= rec_start:
                            overlaps.append(
                                {
                                    "city": record.get("city"),
                                    "dates": f"{rec_start} to {record.get('end_date') or 'Present'}",
                                }
                            )

        # Save current submission
        new_face = {
            "name": display_name if display_name != "Unknown" else normalized_name,
            "encoding": unknown_encodings[0].tolist(),
            "image_url": image_url,
            "city": data.get("city"),
            "start_date": data.get("start_date"),
            "end_date": data.get("end_date"),
            "review_text": data.get("review_text"),
            "submitter_email": data.get("submitter_email"),
        }

        faces_collection.insert_one(new_face)

        if display_name == "Unknown":
            display_name = normalized_name

        # Optional alert logic (kept as-is)
        if data.get("submitter_email") != "anonymous" and display_name != "Unknown":
            send_alert_email(display_name)

        return jsonify(
            {
                "status": "success",
                "match": display_name,
                "image_url": image_url,
                "city": data.get("city"),
                "details": data.get("review_text"),
                "start_date": data.get("start_date"),
                "end_date": data.get("end_date"),
                "history": history,
                "overlaps": overlaps,
            }
        ), 200

    except Exception as e:
        # True server-side error
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/contact-uploader", methods=["POST"])
def contact_uploader():
    try:
        data = request.get_json(silent=True) or {}
        target_name = data.get("target_name")
        message_content = data.get("message")

        if not target_name or not message_content:
            return jsonify({"status": "error", "message": "target_name and message are required."}), 400

        original_record = faces_collection.find_one({"name": target_name})
        if not original_record or "submitter_email" not in original_record:
            return jsonify({"status": "error", "message": "Uploader contact info unavailable."}), 404

        dest_email = original_record["submitter_email"]

        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = dest_email
        msg["Subject"] = f"Checkmate Inquiry: {target_name}"

        body = (
            f"Hello,\n\nA user has found a match for '{target_name}' and wishes to connect.\n\n"
            f"Message from the user:\n--------------------------------------------------\n"
            f"{message_content}\n--------------------------------------------------\n\n"
            f"To respond, you may reply directly to this email."
        )

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()

        return jsonify({"status": "success", "message": "Inquiry sent privately."}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # IBM Code Engine typically uses PORT=8080
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))