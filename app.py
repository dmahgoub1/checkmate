from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import os
import numpy as np

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# --- DATABASE CONNECTION ---
# This ensures it looks for the Render variable 'MONGO_URI'
MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    print("CRITICAL ERROR: MONGO_URI environment variable not found!")
    # Fallback is removed to prevent it from looking at 'localhost'
else:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client['checkmate_db']
    collection = db['subjects']

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/results.html')
def serve_results():
    return send_from_directory('.', 'results.html')

@app.route('/models/<path:filename>')
def serve_models(filename):
    return send_from_directory('models', filename)

@app.route('/submit-and-check', methods=['POST', 'OPTIONS'])
def submit_and_check():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
        
    data = request.json
    new_descriptor = np.array(data['descriptor'])
    
    match_found = False
    threshold = 0.6 
    all_names, all_cities, all_dates, observations = [], [], [], []

    try:
        # Search MongoDB
        for entry in collection.find():
            old_descriptor = np.array(entry['descriptor'])
            dist = np.linalg.norm(new_descriptor - old_descriptor)
            
            if dist < threshold:
                match_found = True
                if entry.get('name') not in all_names: all_names.append(entry.get('name'))
                if entry.get('city') not in all_cities: all_cities.append(entry.get('city'))
                dr = f"{entry.get('start', 'N/A')} to {entry.get('end', 'N/A')}"
                all_dates.append(dr)
                observations.append({"date": dr, "city": entry.get('city'), "text": entry.get('review_text')})

        # Save the new entry
        collection.insert_one(data)

        return jsonify({
            "status": "match" if match_found else "new",
            "all_names": all_names,
            "cities": all_cities,
            "all_dates": all_dates,
            "observations": observations
        })
    except Exception as e:
        print(f"Database Error: {e}")
        return jsonify({"error": "Database connection failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)