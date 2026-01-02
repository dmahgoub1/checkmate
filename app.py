import os
import numpy as np
import base64
import face_recognition
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient

# --- INITIALIZATION ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- DATABASE CONNECTION ---
# Ensure MONGO_URI is set in your Render Environment Variables
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.checkmate_db
subjects_col = db.subjects
notifications_col = db.notifications 
watchers_col = db.watchers 

def calculate_distance(desc1, desc2):
    """Calculates Euclidean distance between two face encodings."""
    return np.linalg.norm(np.array(desc1) - np.array(desc2))

# --- PAGE ROUTES ---

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/search.html')
def serve_search():
    return send_from_directory(app.static_folder, 'search.html')

@app.route('/results.html')
def serve_results():
    return send_from_directory(app.static_folder, 'results.html')

# --- API ENDPOINTS ---

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    data = request.json
    image_b64 = data.get("image_data")
    
    if not image_b64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # 1. Decode the Base64 image sent from the frontend
        header, encoded = image_b64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = face_recognition.load_image_file(BytesIO(image_bytes))
        
        # 2. Extract Face Encoding (The AI heavy lifting)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) == 0:
            return jsonify({"status": "no_face", "message": "No face detected by server."}), 200
        
        new_descriptor = face_encodings[0]
        
        # 3. Search Database for matches
        all_subjects = list(subjects_col.find({}))
        match_found = None
        threshold = 0.55 # Distance threshold for a match

        for subject in all_subjects:
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
            "image": image_b64, # Save the image with the sighting
            "submitter": data.get("submitter_email", "anonymous")
        }

        if match_found:
            # Add new sighting to existing person
            subjects_col.update_one(
                {"_id": match_found["_id"]},
                {"$push": {"observations": current_sighting}}
            )
            return jsonify({
                "status": "match", 
                "observations": match_found["observations"] + [current_sighting]
            })
        else:
            # Create a brand new record
            subjects_col.insert_one({
                "name": data.get("name"),
                "descriptor": new_descriptor.tolist(),
                "observations": [current_sighting]
            })
            return jsonify({
                "status": "no_match", 
                "observations": [current_sighting]
            })

    except Exception as e:
        print(f"Backend Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/add-to-watchlist', methods=['POST'])
def add_to_watchlist():
    """Saves notification preference for verified users."""
    data = request.json
    watchers_col.update_one(
        {"email": data.get("email")},
        {
            "$addToSet": {"watching_names": data.get("subject_name")},
            "$set": {"notify_by_email": True} # Always include email option
        },
        upsert=True
    )
    return jsonify({"status": "success", "message": "Email notification enabled."})

@app.route('/notify-submitter', methods=['POST'])
def notify_submitter():
    """Logs a contact request and ensures email option is preserved."""
    data = request.json
    notifications_col.insert_one({
        "timestamp": datetime.now(),
        "requester_email": data.get("requester_email"),
        "subject_name": data.get("subject_name"),
        "message": data.get("message"),
        "status": "unread",
        "email_notification_active": True # User preference: Never remove
    })
    return jsonify({"message": "Notification logged successfully"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)