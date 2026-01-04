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
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/checkmate")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
ALERTS_EMAIL = os.environ.get("ALERTS_EMAIL")

# --- DATABASE SETUP ---
client = MongoClient(MONGO_URI)
db = client.get_default_database()
faces_collection = db.known_faces

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

@app.route('/')
def index(): return render_template('index.html')

@app.route('/search')
def search_page(): return render_template('search.html')

@app.route('/results')
def results_page(): return render_template('results.html')

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    try:
        data = request.get_json()
        # Normalize the input name to lowercase
        raw_name = data.get('name', 'Unknown')
        normalized_name = raw_name.lower().strip() 

        header, encoded = data['image_data'].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        unknown_encodings = face_recognition.face_encodings(rgb_img)
        if not unknown_encodings: return jsonify({"status": "no_face_detected"}), 200

        known_faces_data = list(faces_collection.find())
        display_name = "Unknown"; history = []; overlaps = []

        # 1. SEARCH FOR EXISTING MATCH
        if known_faces_data:
            known_encs = [np.array(f['encoding']) for f in known_faces_data]
            # Use the stored names (which are now lowercase)
            known_names = [f['name'] for f in known_faces_data]
            matches = face_recognition.compare_faces(known_encs, unknown_encodings[0])
            
            if True in matches:
                display_name = known_names[matches.index(True)]
                # Search history using lowercase name
                history = list(faces_collection.find({"name": display_name}, {"_id": 0, "encoding": 0, "submitter_email": 0}))
                
                user_start = data.get('start_date')
                user_end = data.get('end_date') or "9999-12-31"
                
                for record in history:
                    rec_start = record.get('start_date')
                    rec_end = record.get('end_date') or "9999-12-31"
                    if user_start and rec_start:
                        if user_start <= rec_end and user_end >= rec_start:
                            overlaps.append({"city": record.get('city'), "dates": f"{rec_start} to {record.get('end_date') or 'Present'}"})

        # 2. SAVE AS NEW RECORD IF NOT FOUND
        if display_name == "Unknown":
            display_name = normalized_name
            new_face = {
                "name": normalized_name, # Saved as lowercase
                "encoding": unknown_encodings[0].tolist(),
                "city": data.get('city'),
                "start_date": data.get('start_date'),
                "end_date": data.get('end_date'),
                "review_text": data.get('review_text'),
                "submitter_email": data.get('submitter_email')
            }
            faces_collection.insert_one(new_face)

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


@app.route('/contact-uploader', methods=['POST'])
def contact_uploader():
    """Sends a private message to an uploader without revealing their email to the sender."""
    try:
        data = request.get_json()
        target_name = data.get('target_name')
        message_content = data.get('message')
        
        # Internal lookup for the uploader's email
        original_record = faces_collection.find_one({"name": target_name})
        
        if not original_record or 'submitter_email' not in original_record:
            return jsonify({"status": "error", "message": "Uploader contact info unavailable."}), 404

        dest_email = original_record['submitter_email']
        
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = dest_email
        msg['Subject'] = f"Checkmate Inquiry: {target_name}"
        
        body = (f"Hello,\n\nA user has found a match for '{target_name}' and wishes to connect.\n\n"
                f"Message from the user:\n"
                f"--------------------------------------------------\n"
                f"{message_content}\n"
                f"--------------------------------------------------\n\n"
                f"To respond, you may reply directly to this email.")
        
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()

        return jsonify({"status": "success", "message": "Inquiry sent privately."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))