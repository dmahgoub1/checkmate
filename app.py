from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import numpy as np

app = Flask(__name__, static_folder='.') # Tell Flask to look in the root folder
CORS(app)

DB_FILE = 'database.json'

# --- NEW: ADD THESE ROUTES TO FIX THE 404 ---

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# --- YOUR EXISTING LOGIC BELOW ---

def load_db():
    if not os.path.exists(DB_FILE): return []
    try:
        with open(DB_FILE, 'r') as f: return json.load(f)
    except: return []

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

@app.route('/submit-and-check', methods=['POST'])
def submit_and_check():
    data = request.json
    new_descriptor = np.array(data['descriptor'])
    db = load_db()
    
    match_found = False
    threshold = 0.6 
    
    all_names, all_cities, all_dates, observations = [], [], [], []

    for entry in db:
        old_descriptor = np.array(entry['descriptor'])
        dist = np.linalg.norm(new_descriptor - old_descriptor)
        if dist < threshold:
            match_found = True
            if entry.get('name') not in all_names: all_names.append(entry.get('name'))
            if entry.get('city') not in all_cities: all_cities.append(entry.get('city'))
            date_range = f"{entry.get('start', 'N/A')} to {entry.get('end', 'N/A')}"
            all_dates.append(date_range)
            observations.append({"date": date_range, "city": entry.get('city'), "text": entry.get('review_text')})

    db.append(data)
    save_db(db)

    if match_found:
        return jsonify({"status": "match", "all_names": all_names, "cities": all_cities, "all_dates": all_dates, "observations": observations})
    return jsonify({"status": "new"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)