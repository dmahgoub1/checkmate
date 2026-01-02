import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)
CORS(app)

# --- DATABASE CONFIGURATION ---
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects

def calculate_distance(desc1, desc2):
    """Calculates Euclidean distance between two face descriptors."""
    return np.linalg.norm(np.array(desc1) - np.array(desc2))

@app.route('/')
def health_check():
    return "Checkmate API is Live.", 200

@app.route('/submit-and-check', methods=['POST', 'OPTIONS'])
def submit_and_check():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ready"}), 200

    data = request.json
    new_descriptor = data.get("descriptor")
    new_name = data.get("name")
    new_city = data.get("city")
    new_date_context = data.get("date_context")
    new_review = data.get("review_text")
    new_image = data.get("image_data")

    if not new_descriptor or not new_name:
        return jsonify({"error": "Missing required data"}), 400

    all_subjects = list(subjects_col.find({}))
    match_found = None
    threshold = 0.6 

    for subject in all_subjects:
        distance = calculate_distance(new_descriptor, subject['descriptor'])
        if distance < threshold:
            match_found = subject
            break

    if match_found:
        # Create the new observation record
        new_obs = {
            "city": new_city,
            "date": new_date_context,
            "text": new_review,
            "image": new_image
        }
        
        # Add it to the existing subject's history
        subjects_col.update_one(
            {"_id": match_found["_id"]},
            {"$push": {"observations": new_obs}}
        )
        
        # Return history including the original photo
        return jsonify({
            "status": "match",
            "observations": match_found.get("observations", []) + [new_obs]
        })

    else:
        # Create a new subject document
        new_subject = {
            "name": new_name,
            "descriptor": new_descriptor,
            "image_data": new_image,
            "observations": [{
                "city": new_city,
                "date": new_date_context,
                "text": new_review,
                "image": new_image
            }]
        }
        subjects_col.insert_one(new_subject)
        
        return jsonify({
            "status": "no_match",
            "message": "New profile created."
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)