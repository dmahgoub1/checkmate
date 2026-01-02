import os
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

# --- INITIALIZATION ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- DATABASE CONNECTION ---
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects
notifications_col = db.notifications 
watchers_col = db.watchers 

def calculate_distance(desc1, desc2):
    return np.linalg.norm(np.array(desc1) - np.array(desc2))

# --- CRITICAL: AI MODEL SERVICING ---
# This ensures Render sends the .json and .shard files correctly to the browser
@app.route('/models/<path:filename>')
def serve_models(filename):
    return send_from_directory('models', filename)

# --- PAGE ROUTES ---

@app.route('/')
def serve_index():
    # Landing / Login Page
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/search.html')
def serve_search():
    # Face Scan & Upload Page
    return send_from_directory(app.static_folder, 'search.html')

@app.route('/results.html')
def serve_results():
    # History & Contact Page
    return send_from_directory(app.static_folder, 'results.html')

# --- API ENDPOINTS ---

@app.route('/add-to-watchlist', methods=['POST'])
def add_to_watchlist():
    """Saves a user's request to be notified of future matches."""
    data = request.json
    email = data.get("email")
    subject_name = data.get("subject_name")
    
    watchers_col.update_one(
        {"email": email},
        {"$addToSet": {"watching_names": subject_name}},
        upsert=True
    )
    return jsonify({"status": "success", "message": "Added to watchlist"})

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

    # Search for match in database
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
        subjects_col.update