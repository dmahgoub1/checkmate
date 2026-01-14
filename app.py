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
# Suggested improvement: Added timeout and connection verification to prevent silent crashes
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # This triggers a call to the server to verify the connection is alive
    client.server_info() 
    print("Database connected successfully")
except Exception as e:
    print(f"DATABASE CONNECTION ERROR: {e}")

db = client.checkmate
faces_collection = db.known_faces
watch_collection = db.watch_requests # New collection for future notifications

def send_alert_email(name):
    if not all([SMTP_USER, SMTP_PASS, ALERTS_EMAIL]): return
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = ALERTS_EMAIL
    msg['Subject'] = f"ALERT: {name} Identified"
    msg.attach(MIMEText(f"The individual '{name}' has been scanned and identified on Checkmate.", 'plain'))
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Email error: {e}")

def notify_watchers(matched_name):
    watchers = watch_collection.find({"target_name": matched_name})
    for watch in watchers:
        user_email = watch.get('user_email')
        if not user_email: continue
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER; msg['To'] = user_email; msg['Subject'] = f"Checkmate Update: {matched_name} Found"
        body = f"Hello,\n\nYou are receiving this because you 'Watched' {matched_name}. A new record or scan has just been matched to this person."
        msg.attach(MIMEText(body, 'plain'))
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT); server.starttls()
            server.login(SMTP_USER, SMTP_PASS); server.send_message(msg); server.quit()
        except: pass

@app.route('/')
def index():
    return render_template('search.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    try:
        data = request.get_json()
        image_data = data.get('image_data').split(",")[1]
        name = data.get('name')
        city = data.get('city')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        review_text = data.get('review_text')
        submitter_email = data.get('submitter_email', 'anonymous')

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_encodings = face_recognition.face_encodings(rgb_img)
        if not face_encodings:
            return jsonify({"status": "no_face_detected", "message": "No face found in image."}), 200

        new_encoding = face_encodings[0]
        
        # Check against database
        all_records = list(faces_collection.find())
        match_found = False
        matched_name = ""

        for record in all_records:
            db_encoding = np.array(record['encoding'])
            results = face_recognition.compare_faces([db_encoding], new_encoding, tolerance=0.5)
            if results[0]:
                match_found = True
                matched_name = record['name']
                break

        # Save current entry
        new_face = {
            "name": name, "city": city, "start_date": start_date, "end_date": end_date,
            "review_text": review_text, "encoding": new_encoding.tolist(),
            "submitter_email": submitter_email
        }
        faces_collection.insert_one(new_face)

        if match_found:
            send_alert_email(matched_name)
            notify_watchers(matched_name)
            history = list(faces_collection.find({"name": matched_name}, {"_id": 0, "encoding": 0}))
            return jsonify({"status": "success", "match": matched_name, "history": history}), 200
        else:
            return jsonify({"status": "success", "match": None}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/add-watch', methods=['POST'])
def add_watch():
    try:
        data = request.get_json()
        watch_collection.insert_one({
            "target_name": data.get('target_name'),
            "user_email": data.get('user_email')
        })
        return jsonify({"status": "success", "message": "You will be notified of future matches."}), 200
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
        body = (f"Hello,\n\nA user has found a match for '{target_name}' and wishes to connect.\n\n"
                f"Message from the user:\n--------------------------------------------------\n"
                f"{message_content}\n--------------------------------------------------\n\n"
                f"To respond, you may reply directly to this email.")
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT); server.starttls()
        server.login(SMTP_USER, SMTP_PASS); server.send_message(msg); server.quit()
        return jsonify({"status": "success", "message": "Inquiry sent privately."}), 200
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)