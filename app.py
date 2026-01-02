from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import numpy as np
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
CORS(app)

DB_FILE = 'database.json'

# --- EMAIL CONFIG (Optional for alerts) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get("SENDER_EMAIL") 
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD") 

def send_alert_email(target_email, subject_name):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return
    try:
        msg = MIMEText(f"Watchlist Alert: A new search has been performed on '{subject_name}'.")
        msg['Subject'] = f"TRUST YOUR HEART ALERT: {subject_name}"
        msg['From'] = SENDER_EMAIL
        msg['To'] = target_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Email error: {e}")

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
    
    all_names = []
    all_cities = []
    all_dates = []
    observations = []
    notify_list = []

    for entry in db:
        old_descriptor = np.array(entry['descriptor'])
        dist = np.linalg.norm(new_descriptor - old_descriptor)
        
        if dist < threshold:
            match_found = True
            
            # Aggregate match data
            name = entry.get('name', 'Unknown')
            city = entry.get('city', 'Unknown')
            if name not in all_names: all_names.append(name)
            if city not in all_cities: all_cities.append(city)
            
            date_range = f"{entry.get('start', 'N/A')} to {entry.get('end', 'N/A')}"
            all_dates.append(date_range)
            
            # Collect historical comment
            observations.append({
                "date": date_range,
                "city": city,
                "text": entry.get('review_text', 'No notes provided.')
            })
            
            if entry.get('notif_email'):
                notify_list.append(entry['notif_email'])

    # Save new record
    db.append(data)
    save_db(db)

    # Trigger Alerts
    for email in set(notify_list):
        send_alert_email(email, data['name'])

    if match_found:
        return jsonify({
            "status": "match",
            "all_names": all_names,
            "cities": all_cities,
            "all_dates": all_dates,
            "observations": observations
        })
    return jsonify({"status": "new"})

@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.json
    email = data.get('email')
    descriptor = np.array(data.get('descriptor'))
    
    db = load_db()
    # Match against the last entry to attach email
    for entry in reversed(db):
        if np.linalg.norm(np.array(entry['descriptor']) - descriptor) < 0.1:
            entry['notif_email'] = email
            break
            
    save_db(db)
    return jsonify({"status": "subscribed"})

if __name__ == '__main__':
    # Use environment port for Render/Heroku
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)