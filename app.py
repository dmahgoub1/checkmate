# Add this new route to your existing app.py

@app.route('/add-to-watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.json
    email = data.get("email")
    subject_id = data.get("subject_id")
    
    # Stores who is watching which person
    db.watchers.update_one(
        {"email": email},
        {"$addToSet": {"watching_ids": subject_id}},
        upsert=True
    )
    return jsonify({"status": "success"})