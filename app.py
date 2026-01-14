import os
import base64
import smtplib
import numpy as np
import cv2
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
import face_recognition

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# We retrieve variables, ensuring we handle cases where they might be missing
MONGO_URI = os.environ.get("MONGO_URI")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
ALERTS_EMAIL = os.environ.get("ALERTS_EMAIL")

# --- DATABASE SETUP (Defensive) ---
# We wrap this in a try/except so a secret mapping error doesn't kill the app startup
client = None
db = None
faces_collection = None
watch_collection = None

try:
    if not MONGO_URI:
        print("CRITICAL: MONGO_URI environment variable is missing or empty.", flush=True)
    else:
        # Added serverSelectionTimeoutMS so it doesn't hang forever if the URI is wrong
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Verify connection immediately
        client.admin.command('ping')
        
        db = client.checkmate
        faces_collection = db.known_faces
        watch_collection = db.watch_requests
        print("Database connected successfully!", flush=True)
except Exception as e:
    print(f"DATABASE CONNECTION ERROR: {e}", flush=True)
    # We leave client/db as None; the routes will handle this gracefully later

def send_alert_email(name):
    if not all([SMTP_USER, SMTP_PASS, ALERTS_EMAIL]): 
        print(f"Skipping email alert for {name}: SMTP settings incomplete.", flush=True)
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ALERTS_EMAIL
        msg['Subject'] = f"ALERT: {name} Identified"
        body = f"The system has identified a person of interest: {name}.\nCheck the dashboard for details."
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Alert email sent for {name}", flush=True)
    except Exception as e:
        print(f"Failed to send email: {e}", flush=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search_page():
    return render_template('search.html')

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    # Graceful check if database is down
    if faces_collection is None:
        return jsonify({"status": "error", "message": "Database connection not established. Check server logs."}), 500

    try:
        data = request.get_json()
        image_data = data.get('image')
        name = data.get('name')
        email = data.get('email')
        city = data.get('city')
        comments = data.get('comments')

        if not image_data:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        # Decode image
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Face recognition
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_img)

        if not face_encodings:
            return jsonify({"status": "no_face_detected", "message": "No face found in image."}), 200

        current_encoding = face_encodings[0]
        
        # Pull all known faces from DB
        all_records = list(faces_collection.find())
        match_found = False
        matched_name = None

        for record in all_records:
            known_encoding = np.array(record['encoding'])
            results = face_recognition.compare_faces([known_encoding], current_encoding, tolerance=0.6)
            if results[0]:
                match_found = True
                matched_name = record['name']
                break

        if match_found:
            send_alert_email(matched_name)
            return jsonify({
                "status": "success",
                "match": matched_name,
                "message": f"Match found: {matched_name}. Notification sent."
            }), 200
        else:
            # Store as a "watch request" if no match
            watch_collection.insert_one({
                "name": name,
                "email": email,
                "city": city,
                "comments": comments,
                "encoding": current_encoding.tolist()
            })
            return jsonify({"status": "success", "match": None, "message": "No match found. Added to watch list."}), 200

    except Exception as e:
        print(f"Error in submit-and-check: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Use environment port for Code Engine compatibility
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)