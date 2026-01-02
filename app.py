import os
import base64
import requests
import numpy as np
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

client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects

def calculate_distance(desc1, desc2):
    d1 = np.array(desc1)
    d2 = np.array(desc2)
    if d1.shape != d2.shape: return 999.0
    return np.linalg.norm(d1 - d2)

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
        
        if "," in image_b64:
            encoded = image_b64.split(",")[1]
        else:
            encoded = image_b64
        
        image_bytes = base64.b64decode(encoded)

        files = { 'media': ('image.jpg', image_bytes, 'image/jpeg') }
        params = {
            'api_user': SIGHTENGINE_USER,
            'api_secret': SIGHTENGINE_SECRET,
            'models': 'face-attributes'
        }
        
        r = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
        output = r.json()

        if output.get('status') != 'success' or not output.get('faces'):
            return jsonify({"status": "no_face", "message": "No face detected"}), 200

        f = output['faces'][0]['features']
        new_descriptor = [
            f['left_eye']['x'], f['left_eye']['y'],
            f['right_eye']['x'], f['right_eye']['y'],
            f['nose_tip']['x'], f['nose_tip']['y'],
            f['left_mouth_corner']['x'], f['left_mouth_corner']['y'],
            f['right_mouth_corner']['x'], f['right_mouth_corner']['y']
        ]

        all_subjects = list(subjects_col.find({}))
        match_found = None
        threshold = 0.25 

        for subject in all_subjects:
            dist = calculate_distance(new_descriptor, subject['descriptor'])
            if dist < threshold:
                match_found = subject
                break

        # Explicitly handling the submitter info
        submitter = data.get("submitter_email", "anonymous")

        current_sighting = {
            "city": data.get("city") or "Unknown",
            "date": data.get("date_context") or "Unknown",
            "submitted_on": datetime.now().strftime("%b %d, %Y"),
            "text": data.get("review_text", ""),
            "image": image_b64,
            "submitter": submitter
        }

        if match_found:
            subjects_col.update_one({"_id": match_found["_id"]}, {"$push": {"observations": current_sighting}})
            return jsonify({
                "status": "match", 
                "observations": match_found["observations"] + [current_sighting],
                "user_query": {"name": match_found["name"]}
            })
        else:
            subjects_col.insert_one({
                "name": data.get("name") or "Unnamed Subject",
                "descriptor": new_descriptor,
                "observations": [current_sighting]
            })
            return jsonify({
                "status": "no_match", 
                "observations": [current_sighting],
                "user_query": {"name": data.get("name") or "Unnamed Subject"}
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))