import os
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- DATABASE ---
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects
notifications_col = db.notifications 

def calculate_distance(desc1, desc2):
    return np.linalg.norm(np.array(desc1) - np.array(desc2))

# --- PAGE ROUTES ---

@app.route('/')
def serve_index():
    # Landing / Login Page
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/search.html')
def serve_search():
    # Face Scan Page
    return send_from_directory(app.static_folder, 'search.html')

@app.route('/results.html')
def serve_results():
    # History / Contact Page
    return send_from_directory(app.static_folder, 'results.html')

# --- API ENDPOINTS ---

@app.route('/submit-and-check', methods=['POST', 'OPTIONS'])
def submit_and_check():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ready"}), 200

    data = request.json
    new_descriptor = data.get("descriptor")
    new_name = data.get("name")
    new_city = data.get("city")
    new_date_context = data.get("date_context")
    new_review = data.get("review_text", "")
    new_image = data.get("image_data")
    submitter_email = data.get("submitter_email", "anonymous")

    if not new_descriptor or not new_name:
        return jsonify({"error": "Missing data"}), 400

    all_subjects = list(subjects_col.find({}))
    match_found = None
    threshold = 0.6 

    for subject in all_subjects:
        distance = calculate_distance(new_descriptor, subject['descriptor'])
        if distance < threshold:
            match_found = subject
            break

    current_sighting = {
        "city": new_city,
        "date": new_date_context,
        "submitted_on": datetime.now().strftime("%b %d, %Y"),
        "text": new_review,
        "image": new_image,
        "submitter": submitter_email
    }

    if match_found:
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
        new_record = {
            "name": new_name,
            "descriptor": new_descriptor,
            "image_data": new_image,
            "observations": [current_sighting]
        }
        subjects_col.insert_one(new_record)
        return jsonify({
            "status": "no_match", 
            "observations": [current_sighting]
        })

@app.route('/notify-submitter', methods=['POST'])
def notify_submitter():
    data = request.json
    notifications_col.insert_one({
        "timestamp": datetime.now(),
        "requester_email": data.get("requester_email"),
        "subject_name": data.get("subject_name"),
        "message": data.get("message"),
        "status": "unread"
    })
    return jsonify({"message": "Logged successfully"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)