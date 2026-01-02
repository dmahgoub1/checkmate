import os
import requests
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

# --- INITIALIZATION ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- CONFIG (Add these to Render Environment Variables) ---
MONGO_URI = os.environ.get("MONGO_URI")
SIGHTENGINE_USER = os.environ.get("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_SECRET")

client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects
watchers_col = db.watchers

def calculate_distance(desc1, desc2):
    """Calculates how similar two faces are."""
    return np.linalg.norm(np.array(desc1) - np.array(desc2))

# --- PAGE ROUTES ---
@app.route('/')
def serve_index(): return send_from_directory(app.static_folder, 'index.html')

@app.route('/search.html')
def serve_search(): return send_from_directory(app.static_folder, 'search.html')

@app.route('/results.html')
def serve_results(): return send_from_directory(app.static_folder, 'results.html')

# --- CORE SEARCH LOGIC ---
@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    data = request.json
    image_b64 = data.get("image_data")
    
    if not image_b64:
        return jsonify({"error": "No image data"}), 400

    # 1. Ask Sightengine to analyze the face
    # We use the 'face-attributes' model to get a unique feature vector
    params = {
        'api_user': SIGHTENGINE_USER,
        'api_secret': SIGHTENGINE_SECRET,
        'models': 'face-attributes',
        'base64': image_b64
    }
    
    try:
        r = requests.post('https://api.sightengine.com/1.0/check.json', data=params)
        output = r.json()

        if output.get('status') == 'failure':
            print(f"Sightengine Error: {output.get('error')}")
            return jsonify({"error": "AI Service Error"}), 500

        if not output.get('faces'):
            return jsonify({"status": "no_face", "message": "No face detected."}), 200

        # Create a unique descriptor from the facial features provided by the API
        # We use a combination of landmarks/features to create a stable 'fingerprint'
        face = output['faces'][0]
        new_descriptor = [
            face['shape']['eye_left'][0], face['shape']['eye_left'][1],
            face['shape']['eye_right'][0], face['shape']['eye_right'][1],
            face['shape']['nose_tip'][0], face['shape']['nose_tip'][1],
            face['attributes']['female'], face['attributes']['male']
        ]

        # 2. Compare against our MongoDB database
        all_subjects = list(subjects_col.find({}))
        match_found = None
        threshold = 0.15 # Adjusted for Sightengine landmark vectors

        for subject in all_subjects:
            dist = calculate_distance(new_descriptor, subject['descriptor'])
            if dist < threshold:
                match_found = subject
                break

        current_sighting = {
            "city": data.get("city"),
            "date": data.get("date_context"),
            "submitted_on": datetime.now().strftime("%b %d, %Y"),
            "text": data.get("review_text", ""),
            "image": image_b64,
            "submitter": data.get("submitter_email", "anonymous")
        }

        if match_found:
            subjects_col.update_one({"_id": match_found["_id"]}, {"$push": {"observations": current_sighting}})
            return jsonify({
                "status": "match", 
                "observations": match_found["observations"] + [current_sighting]
            })
        else:
            subjects_col.insert_one({
                "name": data.get("name"),
                "descriptor": new_descriptor,
                "observations": [current_sighting]
            })
            return jsonify({"status": "no_match", "observations": [current_sighting]})

    except Exception as e:
        print(f"System Error: {e}")
        return jsonify({"error": "Processing failed"}), 500

@app.route('/add-to-watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.json
    watchers_col.update_one(
        {"email": data.get("email")},
        {"$addToSet": {"watching_names": data.get("subject_name")}, "$set": {"notify_by_email": True}},
        upsert=True
    )
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))