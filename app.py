import os
import base64
import smtplib
import numpy as np
import cv2
import secrets
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
results_cache_collection = None

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
        results_cache_collection = db.results_cache
        print("Database connected successfully!", flush=True)
except Exception as e:
    print(f"DATABASE CONNECTION ERROR: {e}", flush=True)
    # We leave client/db as None; the routes will handle this gracefully later

def send_alert_email(name, city, start_date, end_date, review_text, recipient_email, results_link):
    if not all([SMTP_USER, SMTP_PASS]): 
        print(f"Skipping email alert for {name}: SMTP settings incomplete.", flush=True)
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = recipient_email
        msg['Subject'] = f"ALERT: {name} has been identified on CheckMate"
        
        body = f"""Alert: {name} has been identified in the CheckMate system!

Someone has uploaded a new photo matching this person.

New Activity Details:
- Name: {name}
- City: {city or 'Not specified'}
- Dating Period: {start_date or 'Not specified'} to {end_date or 'Present'}
- Comments: {review_text or 'No comments provided'}

View full history and all photos for {name}:
{results_link}

---
This is an automated alert from CheckMate.
You received this email because you subscribed to watch alerts for {name}.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Alert email sent to {recipient_email} for {name}", flush=True)
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}", flush=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search_page():
    return render_template('search.html')

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/results/<result_id>')
def results_by_id(result_id):
    return render_template('results_shared.html', result_id=result_id)

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/api/results/<result_id>', methods=['GET'])
def get_results(result_id):
    if results_cache_collection is None:
        return jsonify({"status": "error", "message": "Database connection not established."}), 500
    
    try:
        cached_result = results_cache_collection.find_one({"result_id": result_id})
        
        if not cached_result:
            return jsonify({"status": "error", "message": "Results not found or expired."}), 404
        
        # Return the cached results data
        return jsonify({
            "status": "success",
            "match": cached_result.get('match'),
            "history": cached_result.get('history'),
            "start_date": cached_result.get('start_date'),
            "end_date": cached_result.get('end_date'),
            "city": cached_result.get('city')
        }), 200
        
    except Exception as e:
        print(f"Error retrieving results: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/watch', methods=['POST'])
def watch():
    if watch_collection is None:
        return jsonify({"status": "error", "message": "Database connection not established."}), 500
    
    try:
        data = request.get_json()
        target_name = data.get('target_name')
        email = data.get('email')
        
        if not target_name or not email:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
        # Check if watch already exists
        existing = watch_collection.find_one({"target_name": target_name, "email": email})
        if existing:
            return jsonify({"status": "info", "message": "You're already watching this person."}), 200
        
        # Create new watch request
        watch_collection.insert_one({
            "target_name": target_name,
            "email": email
        })
        
        return jsonify({"status": "success", "message": "Watch request created! You'll be notified of future matches."}), 200
        
    except Exception as e:
        print(f"Error in watch route: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    # Graceful check if database is down
    if faces_collection is None or results_cache_collection is None:
        return jsonify({"status": "error", "message": "Database connection not established. Check server logs."}), 500

    try:
        data = request.get_json()
        image = data.get('image')
        name = data.get('name')
        city = data.get('city')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        review_text = data.get('review_text')
        submitter_email = data.get('submitter_email')

        if not image:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        # Decode image
        header, encoded = image.split(",", 1)
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
            # Fetch ALL historical records for this matched person
            history_records = list(faces_collection.find({"name": matched_name}))
            
            # Format history for frontend
            history = []
            for rec in history_records:
                history.append({
                    "image_data": rec.get('image'),
                    "start_date": rec.get('start_date', 'N/A'),
                    "end_date": rec.get('end_date', 'Present'),
                    "city": rec.get('city', 'Unknown'),
                    "review_text": rec.get('review_text', 'No comments provided.')
                })
            
            # Generate unique result ID and cache the results
            result_id = secrets.token_urlsafe(16)
            results_cache_collection.insert_one({
                "result_id": result_id,
                "match": matched_name,
                "history": history,
                "start_date": start_date,
                "end_date": end_date,
                "city": city
            })
            
            # Create shareable link
            results_link = f"https://thecheckmateapp.com/results/{result_id}"
            
            # Send alerts to all watchers for this person
            watchers = list(watch_collection.find({"target_name": matched_name}))
            for watcher in watchers:
                send_alert_email(
                    name=matched_name,
                    city=city,
                    start_date=start_date,
                    end_date=end_date,
                    review_text=review_text,
                    recipient_email=watcher['email'],
                    results_link=results_link
                )
            
            return jsonify({
                "status": "success",
                "match": matched_name,
                "message": f"Match found: {matched_name}. Notification sent.",
                "history": history,
                "start_date": start_date,
                "end_date": end_date,
                "city": city
            }), 200
        else:
            # Store the NEW submission with all details
            faces_collection.insert_one({
                "name": name,
                "email": submitter_email,
                "city": city,
                "start_date": start_date,
                "end_date": end_date,
                "review_text": review_text,
                "image": image,  # Store the image for history
                "encoding": current_encoding.tolist()
            })
            
            return jsonify({
                "status": "success", 
                "match": None, 
                "message": "No match found. Added to database.",
                "start_date": start_date,
                "end_date": end_date,
                "city": city
            }), 200

    except Exception as e:
        print(f"Error in submit-and-check: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Use environment port for Code Engine compatibility
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)