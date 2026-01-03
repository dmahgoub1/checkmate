import os
import base64
import requests
import smtplib
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
        # Extract base64 part
        encoded = image_b64.split(",")[1] if "," in image_b64 else image_b64
        image_bytes = base64.b64decode(encoded)

        match_found = None
        # We loop through all stored subjects to find a face match
        all_subjects = list(subjects_col.find({}))

        for subject in all_subjects:
            ref_encoded = subject['reference_image'].split(",")[1] if "," in subject['reference_image'] else subject['reference_image']
            ref_bytes = base64.b64decode(ref_encoded)
            
            # Using the Dedicated Comparison API
            r = requests.post('https://api.sightengine.com/1.0/check-sync.json', 
                             files={
                                 'media': ('target.jpg', image_bytes),
                                 'reference': ('ref.jpg', ref_bytes)
                             }, 
                             data={
                                 'api_user': SIGHTENGINE_USER, 
                                 'api_secret': SIGHTENGINE_SECRET,
                                 'models': 'face-compare'
                             })
            
            res = r.json()
            # 0.80 is the "Industry Standard" for confirming it's the same person
            if res.get('status') == 'success' and res.get('faces') and res['faces'][0]['similarity'] > 0.80:
                match_found = subject
                break

        current_sighting = {
            "city": data.get("city") or "Unknown",
            "date": data.get("date_context") or "Unknown",
            "submitted_on": datetime.now().strftime("%b %d, %Y"),
            "text": data.get("review_text", ""),
            "image": image_b64,
            "submitter": data.get("submitter_email", "anonymous")
        }

        if match_found:
            # Add new sighting to existing record
            subjects_col.update_one({"_id": match_found["_id"]}, {"$push": {"observations": current_sighting}})
            
            # TRIGGER WATCHLIST ALERTS
            # If someone else is "watching" this person, notify them of the new upload
            watchers = watchers_col.find({"watching_names": match_found["name"]})
            for w in watchers:
                send_email(w['email'], "CHECKMATE: Sighting Alert", 
                           f"The person you are watching ({match_found['name']}) was just uploaded again in {current_sighting['city']}. Check the app for details.")
            
            return jsonify({
                "status": "match", 
                "observations": match_found["observations"] + [current_sighting], 
                "user_query": {"name": match_found["name"]}
            })
        else:
            # Create a new record and use this image as the future Reference Photo
            subjects_col.insert_one({
                "name": data.get("name") or "Unnamed", 
                "reference_image": image_b64, 
                "observations": [current_sighting]
            })
            return jsonify({
                "status": "no_match", 
                "observations": [current_sighting], 
                "user_query": {"name": data.get("name")}
            })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/request-connection', methods=['POST'])
def request_connection():
    data = request.json
    success = send_email(data.get("target_email"), "Checkmate: Someone wants to connect", 
                         f"A user on Checkmate wants to discuss a mutual match. Contact: {data.get('requester_email')}")
    return jsonify({"status": "sent" if success else "failed"})

@app.route('/add-to-watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.json
    watchers_col.update_one(
        {"email": data.get("email")}, 
        {"$addToSet": {"watching_names": data.get("subject_name")}}, 
        upsert=True
    )
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))