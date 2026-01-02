import os
import numpy as np
import base64
import face_recognition
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# DATABASE
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects
watchers_col = db.watchers

def calculate_distance(desc1, desc2):
    return np.linalg.norm(np.array(desc1) - np.array(desc2))

@app.route('/')
def serve_index(): return send_from_directory(app.static_folder, 'index.html')

@app.route('/search.html')
def serve_search(): return send_from_directory(app.static_folder, 'search.html')

@app.route('/results.html')
def serve_results(): return send_from_directory(app.static_folder, 'results.html')

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    data = request.json
    image_b64 = data.get("image_data")
    
    if not image_b64:
        return jsonify({"error": "No image data"}), 400

    try:
        # 1. Process Image on Backend
        header, encoded = image_b64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = face_recognition.load_image_file(BytesIO(image_bytes))
        
        # 2. Extract Face Encoding
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) == 0:
            return jsonify({"status": "no_face", "message": "No face detected in the photo."}), 200
        
        new_descriptor = face_encodings[0]
        
        # 3. Search Database
        all_subjects = list(subjects_col.find({}))
        match_found = None
        threshold = 0.55 # Slightly stricter for backend processing

        for subject in all_subjects:
            # MongoDB stores lists, we need numpy arrays for math
            db_desc = np.array(subject['descriptor'])
            distance = calculate_distance(new_descriptor, db_desc)
            if distance < threshold:
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
            subjects_col.update_one(
                {"_id": match_found["_id"]},
                {"$push": {"observations": current_sighting}}
            )
            return jsonify({"status": "match", "observations": match_found["observations"] + [current_sighting]})
        else:
            subjects_col.insert_one({
                "name": data.get("name"),
                "descriptor": new_descriptor.tolist(),
                "observations": [current_sighting]
            })
            return jsonify({"status": "no_match", "observations": [current_sighting]})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

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