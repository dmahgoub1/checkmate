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
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
ALERTS_EMAIL = os.environ.get("ALERTS_EMAIL")

# --- DATABASE SETUP ---
client = MongoClient(MONGO_URI)
db = client.checkmate
faces_collection = db.known_faces
watch_collection = db.watch_requests # New collection for future notifications

def send_alert_email(name):
    if not all([SMTP_USER, SMTP_PASS, ALERTS_EMAIL]): return
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = ALERTS_EMAIL
    msg['Subject'] = f"ALERT: {name} Identified"
    msg.attach(MIMEText(f"The individual {name} was identified in a scan.", 'plain'))
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT); server.starttls()
        server.login(SMTP_USER, SMTP_PASS); server.send_message(msg); server.quit()
    except Exception as e: print(f"Email error: {e}")

# Helper for the notification feature
def notify_watchers(new_encoding, matched_name):
    if not all([SMTP_USER, SMTP_PASS]): return
    
    # Get all people watching for this identity
    watchers = watch_collection.find()
    for watch in watchers:
        known_enc = np.array(watch['face_encoding'])
        # Use strict tolerance (0.5) to ensure it's the EXACT same person
        match = face_recognition.compare_faces([known_enc], new_encoding, tolerance=0.5)
        
        if match[0]:
            try:
                msg = MIMEMultipart()
                msg['From'] = SMTP_USER
                msg['To'] = watch['email']
                msg['Subject'] = f"Checkmate Update: New match for {matched_name}"
                body = f"Hello,\n\nA new photo has been uploaded that matches the identity of '{matched_name}', who you are watching."
                msg.attach(MIMEText(body, 'plain'))
                server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT); server.starttls()
                server.login(SMTP_USER, SMTP_PASS); server.send_message(msg); server.quit()
            except Exception as e: print(f"Watcher Notification Error: {e}")

@app.route('/')
def index(): return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'], strict_slashes=False)
def search():
    return render_template('search.html')

@app.route('/results', methods=['GET', 'POST'], strict_slashes=False)
def results():
    return render_template('results.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    try:
        data = request.get_json()
        raw_name = data.get('name', 'Unknown')
        normalized_name = raw_name.lower().strip() 

        header, encoded = data['image_data'].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        unknown_encodings = face_recognition.face_encodings(rgb_img)
        if not unknown_encodings: return jsonify({"status": "no_face_detected"}), 200

        current_encoding = unknown_encodings[0]
        known_faces_data = list(faces_collection.find())
        display_name = "Unknown"; history = []; overlaps = []

        if known_faces_data:
            known_encs = [np.array(f['encoding']) for f in known_faces_data]
            known_names = [f['name'] for f in known_faces_data]
            matches = face_recognition.compare_faces(known_encs, current_encoding)
            
            if True in matches:
                display_name = known_names[matches.index(True)]
                history = list(faces_collection.find({"name": display_name}, {"_id": 0, "encoding": 0, "submitter_email": 0}))
                
                user_start = data.get('start_date')
                user_end = data.get('end_date') or "9999-12-31"
                
                for record in history:
                    rec_start = record.get('start_date')
                    rec_end = record.get('end_date') or "9999-12-31"
                    if user_start and rec_start:
                        if user_start <= rec_end and user_end >= rec_start:
                            overlaps.append({"city": record.get('city'), "dates": f"{rec_start} to {record.get('end_date') or 'Present'}"})

        # Feature: Notify anyone watching for this unique face
        if display_name != "Unknown":
            notify_watchers(current_encoding, display_name)

        # Save current submission to database
        new_face = {
            "name": display_name if display_name != "Unknown" else normalized_name,
            "encoding": current_encoding.tolist(),
            "image_data": data['image_data'],
            "city": data.get('city'),
            "start_date": data.get('start_date'),
            "end_date": data.get('end_date'),
            "review_text": data.get('review_text'),
            "submitter_email": data.get('submitter_email')
        }
        faces_collection.insert_one(new_face)
        if display_name == "Unknown": display_name = normalized_name

        if data.get('submitter_email') != "anonymous" and display_name != "Unknown": 
            send_alert_email(display_name)

        return jsonify({
            "status": "success",
            "match": display_name,
            "city": data.get('city'),
            "details": data.get('review_text'),
            "start_date": data.get('start_date'),
            "end_date": data.get('end_date'),
            "history": history,
            "overlaps": overlaps
        }), 200
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/watch', methods=['POST'])
def add_watch_request():
    try:
        data = request.get_json()
        target_name = data.get('target_name')
        email = data.get('email')
        
        # Get the unique encoding for the specific person the user just saw
        original_record = faces_collection.find_one({"name": target_name})
        if not original_record:
            return jsonify({"status": "error", "message": "Face data not found."}), 404
            
        watch_collection.insert_one({
            "target_name": target_name,
            "email": email,
            "face_encoding": original_record['encoding'] # Store unique identity
        })
        return jsonify({"status": "success", "message": f"You will be notified if {target_name} is uploaded again."}), 200
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/contact-uploader', methods=['POST'])
def contact_uploader():
    try:
        data = request.get_json()
        target_name = data.get('target_name')
        message_content = data.get('message')
        original_record = faces_collection.find_one({"name": target_name})
        if not original_record or 'submitter_email' not in original_record:
            return jsonify({"status": "error", "message": "Uploader contact info unavailable."}), 404
        dest_email = original_record['submitter_email']
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER; msg['To'] = dest_email; msg['Subject'] = f"Checkmate Inquiry: {target_name}"
        body = (f"Hello,\\n\\nA user has found a match for '{target_name}' and wishes to connect.\\n\\n"
                f"Message from the user:\\n--------------------------------------------------\\n"
                f"{message_content}\\n--------------------------------------------------\\n\\n"
                f"To respond, you may reply directly to this email.")
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT); server.starttls()
        server.login(SMTP_USER, SMTP_PASS); server.send_message(msg); server.quit()
        return jsonify({"status": "success", "message": "Inquiry sent privately."}), 200
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

app.config['DEBUG'] = True