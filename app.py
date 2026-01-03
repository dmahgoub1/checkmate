import os
import base64
import requests
import smtplib
import numpy as np
from email.mime.text import MIMEText
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- CONFIG ---
MONGO_URI = os.environ.get("MONGO_URI")
SIGHTENGINE_USER = os.environ.get("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_SECRET")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")

client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects
watchers_col = db.watchers

def send_email(to_email, subject, body):
    if not EMAIL_USER or not EMAIL_PASS:
        print("Email credentials missing.")
        return False
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

def calculate_distance(desc1, desc2):
    d1 = np.array(desc1); d2 = np.array(desc2)
    return np.linalg.norm(d1 - d2) if d1.shape == d2.shape else 999.0

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
        encoded = image_b64.split(",")[1] if "," in image_b64 else image_b64
        image_bytes = base64.b64decode(encoded)

        r = requests.post('https://api.sightengine.com/1.0/check.json', 
                         files={'media': ('image.jpg', image_bytes, 'image/jpeg')}, 
                         data={'api_user': SIGHTENGINE_USER, 'api_secret': SIGHTENGINE_SECRET, 'models': 'face-attributes'})
        output = r.json()

        if output.get('status') != 'success' or not output.get('faces'):
            return jsonify({"status": "no_face"}), 200

        f = output['faces'][0]['features']
        new_descriptor = [f['left_eye']['x'], f['left_eye']['y'], f['right_eye']['x'], f['right_eye']['y'],
                          f['nose_tip']['x'], f['nose_tip']['y'], f['left_mouth_corner']['x'], f['left_mouth_corner']['y'],
                          f['right_mouth_corner']['x'], f['right_mouth_corner']['y']]

        match_found = None
        for subject in subjects_col.find({}):
            if calculate_distance(new_descriptor, subject['descriptor']) < 0.25:
                match_found = subject; break

        current_sighting = {
            "city": data.get("city") or "Unknown",
            "date": data.get("date_context") or "Unknown",
            "submitted_on": datetime.now().strftime("%b %d, %Y"),
            "text": data.get("review_text", ""),
            "image": image_b64,
            "submitter": data.get("submitter_email", "anonymous")
        }

        if match_found:
            subjects_col.update_one({"_id": match_found["_id"]}, {"$push": {"observations": current_sighting}})
            watchers = watchers_col.find({"watching_names": match_found["name"]})
            for w in watchers:
                send_email(w['email'], f"CHECKMATE ALERT: {match_found['name']}", f"A new report was added for {match_found['name']} in {current_sighting['city']}.")
            
            return jsonify({"status": "match", "observations": match_found["observations"] + [current_sighting], "user_query": {"name": match_found["name"]}})
        else:
            subjects_col.insert_one({"name": data.get("name") or "Unnamed", "descriptor": new_descriptor, "observations": [current_sighting]})
            return jsonify({"status": "no_match", "observations": [current_sighting], "user_query": {"name": data.get("name")}})

    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/request-connection', methods=['POST'])
def request_connection():
    data = request.json
    success = send_email(data.get("target_email"), "Checkmate Connection Request", 
                         f"A user wants to connect regarding your post on Checkmate. Reach them at: {data.get('requester_email')}")
    return jsonify({"status": "sent" if success else "failed"})

@app.route('/add-to-watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.json
    watchers_col.update_one({"email": data.get("email")}, {"$addToSet": {"watching_names": data.get("subject_name")}}, upsert=True)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))