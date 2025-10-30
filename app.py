# app.py

import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import numpy as np

import database
import utils

# --- App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key'
DISTANCE_THRESHOLD = 1.1

# --- App Initialization ---
with app.app_context():
    # This now connects to the Milvus server instead of loading local files
    database.init_milvus()


# --- API Endpoints ---

@app.route('/api/register', methods=['POST'])
def api_register():
    image_files = request.files.getlist("image")

    if 'name' not in request.form or not request.form['name']:
        return jsonify({"error": "Missing name"}), 400

    if not 2 <= len(image_files) <= 4:
        return jsonify({"error": f"Please upload between 2 and 4 images. You uploaded {len(image_files)}."}), 400

    name = request.form['name']
    embeddings = []

    for image_file in image_files:
        if image_file.filename == '': continue

        image_bytes = image_file.read()
        embedding = utils.get_face_embedding(image_bytes)

        if embedding is None:
            return jsonify({"error": f"No face detected in one of the images ('{image_file.filename}')."}), 400

        embeddings.append(embedding)

    if len(embeddings) < 2:
        return jsonify({"error": "Fewer than 2 valid images were processed."}), 400

    avg_embedding = np.mean(embeddings, axis=0)
    user_uuid, user_id_human = database.add_user(name, avg_embedding)

    return jsonify({
        "success": True,
        "message": f"User '{name}' registered successfully with ID {user_id_human}.",
        "uuid": user_uuid,
        "human_id": user_id_human
    }), 201


@app.route('/api/authenticate', methods=['POST'])
def api_authenticate():
    if 'image' not in request.files or not request.files['image']:
        return jsonify({"error": "Missing image"}), 400

    embedding = utils.get_face_embedding(request.files['image'].read())
    if embedding is None:
        return jsonify({"authenticated": False, "message": "No face detected"}), 200

    user_uuid, name, human_id, distance = database.find_similar_user(embedding, DISTANCE_THRESHOLD)

    if name:
        return jsonify({
            "authenticated": True,
            "message": f"Authentication successful. Welcome, {name}. Your ID is {human_id}.",
            "uuid": user_uuid, "name": name, "human_id": human_id, "distance": float(distance)
        }), 200
    else:
        return jsonify({"authenticated": False, "message": "User not recognized"}), 200


# --- Web Interface Routes ---

@app.route('/')
def index():
    users = database.get_all_users()
    return render_template('index.html', users=users)


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        response, status_code = api_register()
        response_data = response.get_json()
        if status_code == 201:
            flash(response_data.get('message', 'Registration successful!'), 'success')
        else:
            flash(response_data.get('error', 'An unknown error occurred.'), 'danger')
        return redirect(url_for('index'))
    return render_template('register.html')


@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate_page():
    if request.method == 'POST':
        response, status_code = api_authenticate()
        result = response.get_json()
        if result.get('authenticated'):
            flash(result.get('message', 'Authentication Successful!'), 'success')
        else:
            flash(result.get('message', 'Authentication Failed.'), 'danger')
        return redirect(url_for('index'))
    return render_template('authenticate.html')


@app.route('/user/delete/<user_uuid>', methods=['POST'])
def delete_user_page(user_uuid):
    success = database.delete_user(user_uuid)
    if success:
        flash("User deleted successfully.", "success")
    else:
        flash("Error: User not found.", "danger")
    return redirect(url_for('index'))


# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)