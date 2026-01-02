import os
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

# Initialize Flask and allow it to serve the root directory and /models
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- DATABASE CONFIGURATION ---
# Render will pull the MONGO_URI from your environment variables
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.checkmate_db

# Collections
subjects_col = db.subjects
notifications_col = db.notifications # Stores contact requests and messages

def calculate_distance(desc1, desc2):
    """Calculates Euclidean distance between face descriptors."""
    return np.linalg.norm(np.array(desc1) - np.array(desc2))

# --- ROUTES TO SERVE THE FRONTEND ---

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/results.html')
def serve_results():
    return send_from_directory(app.static_folder, 'results.html')

# --- API ENDPOINTS ---

@app.route('/submit-and-check', methods=['POST', 'OPTIONS'])
def submit_and_check():
    """Handles new photo submissions and database matching."""
    if request.method == 'OPTIONS':
        return jsonify({"status": "ready"}), 200

    data = request.json
    new_descriptor = data.get("descriptor")
    new_name = data.get("name")
    new_city = data.get("city")
    new_date_context = data.get("date_context")
    new_review = data.get("review_text", "")
    new_image = data.get("image_data")
    
    # Placeholder for submitter identity (could be linked to a login later)
    submitter_email = data.get("submitter_email", "anonymous@thecheckmateapp.com")

    if not new_descriptor or not new_name:
        return jsonify({"error": "Missing Photo or Name data"}), 400

    # Fetch all subjects for comparison
    all_subjects = list(subjects_col.find({}))
    match_found = None
    threshold = 0.6 

    for subject in all_subjects:
        distance = calculate_distance(new_descriptor, subject['descriptor'])
        if distance < threshold:
            match_found = subject
            break

    # Create the record for this specific sighting with a server-side timestamp
    current_sighting = {
        "city": new_city,
        "date": new_date_context,
        "submitted_on": datetime.now().strftime("%B %d, %Y"), # e.g., January 02, 2026
        "text": new_review,
        "image": new_image,
        "submitter": submitter_email
    }

    if match_found:
        # Update existing record with the new sighting
        subjects_col.update_one(
            {"_id": match_found["_id"]},
            {"$push": {"observations": current_sighting}}
        )
        updated_subject = subjects_col.find_one({"_id": match_found["_id"]})
        return jsonify({
            "status": "match",
            "observations": updated_subject.get("observations", [])
        })

    else:
        # Create brand new subject profile
        new_record = {
            "name": new_name,
            "descriptor": new_descriptor,
            "image_data": new_image,
            "observations": [current_sighting]
        }
        subjects_col.insert_one(new_record)
        return jsonify({
            "status": "no_match", 
            "message": "New profile created.",
            "observations": [current_sighting]
        })

@app.route('/notify-submitter', methods=['POST'])
def notify_submitter():
    """Logs an anonymous contact request in MongoDB."""
    data = request.json
    
    # This logs the request for you to review in MongoDB Atlas
    notification_entry = {
        "timestamp": datetime.now(),
        "requester_email": data.get("requester_email"),
        "subject_name": data.get("subject_name"),
        "message": data.get("message"), # The reason they want to connect
        "status": "unread"
    }
    
    notifications_col.insert_one(notification_entry)
    
    return jsonify({"message": "Your request has been logged and the submitter will be notified."}), 200

if __name__ == '__main__':
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)