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
MODELS_DIR = "models" # Your folder containing reference photos

client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects
watchers_col = db.watchers

# --- LOCAL AI LOADING ---
known_face_encodings = []
known_face_names = []

def load_local_models():
    """Loads all images from the models folder and encodes faces into memory"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created {MODELS_DIR} folder. Add .jpg files here to seed the AI.")
        return

    for filename in os.listdir(MODELS_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(MODELS_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                # Name is the filename without extension
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"Loaded model: {filename}")

# Run the loader at startup
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
        
        # 1. Convert Base64 to Image for local AI
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        image_bytes = base64.b64decode(encoded)
        img_file = BytesIO(image_bytes)
        unknown_image = face_recognition.load_image_file(img_file)

        # 2. Get encoding for the uploaded photo
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        
        if len(unknown_encodings) == 0:
            return jsonify({"error": "No face detected in the photo. Try again."}), 400
        
        uploaded_encoding = unknown_encodings[0]

        # 3. Compare uploaded face against our 'models' folder in memory
        matches = face_recognition.compare_faces(known_face_encodings, uploaded_encoding, tolerance=0.6)
        match_name = None

        if True in matches:
            first_match_index = matches.index(True)
            match_name = known_face_names[first_match_index]

        # 4. Prepare Sighting Data
        current_sighting = {
            "city": data.get("city") or "Unknown",
            "date": data.get("date_context") or "Unknown",
            "submitted_on": datetime.now().strftime("%b %d, %Y"),
            "text": data.get("review_text", ""),
            "image": image_b64,
            "submitter": data.get("submitter_email", "anonymous")
        }

        # 5. Database Logic (Match vs New)
        if match_name:
            # Look up the match in MongoDB by the name found by the AI
            match_data = subjects_col.find_one({"name": match_name})
            
            if match_data:
                subjects_col.update_one({"_id": match_data["_id"]}, {"$push": {"observations": current_sighting}})
                
                # TRIGGER ALERTS
                watchers = watchers_col.find({"watching_names": match_name})
                for w in watchers:
                    send_email(w['email'], f"CHECKMATE ALERT: {match_name}", 
                               f"A new sighting was reported for {match_name} in {current_sighting['city']}.")
                
                return jsonify({
                    "status": "match", 
                    "observations": match_data["observations"] + [current_sighting], 
                    "user_query": {"name": match_name}
                })

        # If no match or match not in DB, create new entry
        new_name = data.get("name") or f"Subject_{datetime.now().strftime('%H%M%S')}"
        subjects_col.insert_one({
            "name": new_name, 
            "reference_image": image_b64, 
            "observations": [current_sighting]
        })
        
        # Optionally save this new image to the models folder for future scans
        # with open(f"models/{new_name}.jpg", "wb") as f:
        #     f.write(image_bytes)

        return jsonify({
            "status": "no_match", 
            "observations": [current_sighting], 
            "user_query": {"name": new_name}
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

# Watchlist endpoints
@app.route('/add-to-watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.json
    watchers_col.update_one({"email": data.get("email")}, {"$addToSet": {"watching_names": data.get("subject_name")}}, upsert=True)
    return jsonify({"status": "success"})

@app.route('/request-connection', methods=['POST'])
def request_connection():
    data = request.json
    success = send_email(data.get("target_email"), "Checkmate Connection Request", 
                         f"A user wants to compare notes. Reach them at: {data.get('requester_email')}")
    return jsonify({"status": "sent" if success else "failed"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))