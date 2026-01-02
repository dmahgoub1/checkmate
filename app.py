import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

# Initialize Flask to serve your HTML and the /models folder
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- DATABASE CONFIGURATION ---
# Ensure MONGO_URI is set in your Render Environment Variables
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects

def calculate_distance(desc1, desc2):
    """Calculates Euclidean distance between face descriptors."""
    return np.linalg.norm(np.array(desc1) - np.array(desc2))

# --- ROUTES TO SERVE THE WEBSITE ---

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/results.html')
def serve_results():
    return send_from_directory(app.static_folder, 'results.html')

# --- API ENDPOINT ---

@app.route('/submit-and-check', methods=['POST', 'OPTIONS'])
def submit_and_check():
    # Handle the 'pre-flight' request from the browser
    if request.method == 'OPTIONS':
        return jsonify({"status": "ready"}), 200

    data = request.json
    new_descriptor = data.get("descriptor")
    new_name = data.get("name")
    new_city = data.get("city")
    new_date_context = data.get("date_context")
    new_review = data.get("review_text") # This is now optional
    new_image = data.get("image_data")

    if not new_descriptor or not new_name:
        return jsonify({"error": "Missing Photo or Name data"}), 400

    # Fetch all known subjects from MongoDB
    all_subjects = list(subjects_col.find({}))
    match_found = None
    threshold = 0.6  # Standard sensitivity for face-api.js

    for subject in all_subjects:
        distance = calculate_distance(new_descriptor, subject['descriptor'])
        if distance < threshold:
            match_found = subject
            break

    if match_found:
        # Create the new entry for the history (observations)
        new_obs = {
            "city": new_city,
            "date": new_date_context,
            "text": new_review if new_review else "", # Ensure it's never 'null'
            "image": new_image
        }
        
        # Add this new sighting to the existing person's record
        subjects_col.update_one(
            {"_id": match_found["_id"]},
            {"$push": {"observations": new_obs}}
        )
        
        # Return all history including the newest one
        return jsonify({
            "status": "match",
            "observations": match_found.get("observations", []) + [new_obs]
        })

    else:
        # Create a brand new subject profile
        new_subject = {
            "name": new_name,
            "descriptor": new_descriptor,
            "image_data": new_image,
            "observations": [{
                "city": new_city,
                "date": new_date_context,
                "text": new_review if new_review else "",
                "image": new_image
            }]
        }
        subjects_col.insert_one(new_subject)
        return jsonify({"status": "no_match", "message": "New profile created."})

if __name__ == '__main__':
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)