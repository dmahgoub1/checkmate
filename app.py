import os
import base64
import face_recognition
import numpy as np
import smtplib
from io import BytesIO
from PIL import Image
from email.mime.text import MIMEText
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- CONFIG ---
MONGO_URI = os.environ.get("MONGO_URI")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
MODELS_DIR = "models"

client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects
watchers_col = db.watchers

known_face_encodings = []
known_face_names = []

def load_local_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        return
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(MODELS_DIR, filename)
            # RAM SAVE: Resize models at startup too
            img = Image.open(path).convert('RGB')
            img.thumbnail((600, 600))
            image_array = np.array(img)
            encodings = face_recognition.face_encodings(image_array)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

load_local_models()

def send_email(to_email, subject, body):
    if not EMAIL_USER or not EMAIL_PASS: return False
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

@app.route('/')
def serve_index(): return send_from_directory(app.static_folder, 'index.html')

@app.route('/search.html')
def serve_search(): return send_from_directory(app.static_folder, 'search.html')

@app.route('/results.html')
def serve_results(): return send_from_directory(app.static_folder, 'results.html')

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    try:
        data = request.json
        image_b64 = data.get("image_data")
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        image_bytes = base64.b64decode(encoded)

        # RAM SAVE: Resize the uploaded image immediately
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img.thumbnail((600, 600)) # Reduce size to save memory
        unknown_image = np.array(img)

        unknown_encodings = face_recognition.face_encodings(unknown_image)
        if len(unknown_encodings) == 0:
            return jsonify({"error": "No face detected"}), 400
        
        uploaded_encoding = unknown_encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, uploaded_encoding, tolerance=0.6)
        match_name = None

        if True in matches:
            match_name = known_face_names[matches.index(True)]

        current_sighting = {
            "city": data.get("city") or "Unknown",
            "date": data.get("date_context") or "Unknown",
            "submitted_on": datetime.now().strftime("%b %d, %Y"),
            "text": data.get("review_text", ""),
            "image": image_b64,
            "submitter": data.get("submitter_email", "anonymous")
        }

        if match_name:
            match_data = subjects_col.find_one({"name": match_name})
            if match_data:
                subjects_col.update_one({"_id": match_data["_id"]}, {"$push": {"observations": current_sighting}})
                watchers = watchers_col.find({"watching_names": match_name})
                for w in watchers:
                    send_email(w['email'], f"CHECKMATE ALERT: {match_name}", f"New sighting in {current_sighting['city']}.")
                return jsonify({"status": "match", "observations": match_data["observations"] + [current_sighting], "user_query": {"name": match_name}})

        new_name = data.get("name") or f"Subject_{datetime.now().strftime('%H%M%S')}"
        subjects_col.insert_one({"name": new_name, "reference_image": image_b64, "observations": [current_sighting]})
        return jsonify({"status": "no_match", "observations": [current_sighting], "user_query": {"name": new_name}})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # IBM Code Engine uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)