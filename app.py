import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import face_recognition
import numpy as np
import cv2

app = Flask(__name__)

# --- CONFIGURATION (via Environment Variables) ---
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
    """Sends an email notification when a face match is found."""
    if not all([SMTP_USER, SMTP_PASS, ALERTS_EMAIL]):
        print("Email credentials missing. Skipping alert.")
        return

    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = ALERTS_EMAIL
    msg['Subject'] = f"ALERT: {name} Identified"
    
    body = f"The Checkmate Security system has identified: {name}."
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Alert email sent for {name}")
    except Exception as e:
        print(f"Email failed: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Receives image data, compares with DB, and triggers alerts."""
    file = request.files['image']
    email_enabled = request.form.get('email_enabled') == 'true'
    
    # Load the uploaded image
    img = face_recognition.load_image_file(file)
    unknown_encodings = face_recognition.face_encodings(img)

    if not unknown_encodings:
        return jsonify({"status": "no_face_detected"})

    # Fetch known faces from MongoDB
    known_faces_data = list(faces_collection.find())
    known_encodings = [np.array(f['encoding']) for f in known_faces_data]
    known_names = [f['name'] for f in known_faces_data]

    # Compare
    matches = face_recognition.compare_faces(known_encodings, unknown_encodings[0])
    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]
        
        if email_enabled:
            send_alert_email(name)

    return jsonify({"status": "success", "match": name})

if __name__ == '__main__':
    # Using 8080 for IBM Cloud compatibility
    app.run(host='0.0.0.0', port=8080)