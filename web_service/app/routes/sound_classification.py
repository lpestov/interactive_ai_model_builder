from flask import Blueprint, jsonify, redirect, render_template, request, url_for, flash
from dotenv import load_dotenv
import os
import shutil
import json
import subprocess

from app.exceptions.folderNotFoundError import FolderNotFoundError

load_dotenv()

ALLOWED_EXTENSIONS = {'wav'}
UPLOAD_FOLDER_PATH = 'sounds/sound_dataset/train'
UTILS_PATH = 'utils/sound_classification'
YC_TOKEN = os.environ.get('YC_TOKEN')
PROJECT_ID = os.environ.get('PROJECT_ID')
CONFIG_YAML_FILE = os.environ.get('CONFIG_YAML_FILE')

def file_extension_validation(filename):
    return "." in filename and filename.split(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def create_zip(source_folder, zip_name):
    if not os.path.exists(source_folder):
        raise FolderNotFoundError(f"Folder {source_folder} not found")
    
    shutil.make_archive(os.path.join(UTILS_PATH, zip_name), 'zip', source_folder)

sound_classification_bp = Blueprint('sound_classification', __name__)

@sound_classification_bp.route('/sound_classification', methods=['GET'])
def index():
    return render_template('sound_classification.html', active_page='sound')

@sound_classification_bp.route('/upload_sounds', methods=['POST', 'GET'])
def upload_sounds():
    os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)

    if not request.files:
        flash('No downloaded files')
        return redirect(url_for('sound_classification.index'))

    class_file_count = {}

    for class_id in request.files:
        class_name = class_id.replace("[]", "").lower()
        files = request.files.getlist(class_id)
        valid_files = [f for f in files if f.filename and file_extension_validation(f.filename)]

        if class_name not in class_file_count:
            class_file_count[class_name] = 0
        class_file_count[class_name] += len(valid_files)

    invalid_classes = {name: count for name, count in class_file_count.items()
                      if count < 15}

    if invalid_classes:
        error_message = "Not enough audio files for classes: "
        error_message += ", ".join([f"{name} ({count}/15)" for name, count in invalid_classes.items()])
        flash(error_message)
        return redirect(url_for('sound_classification.index'))

    class_mapping = {}
    class_index = 0

    for class_id in request.files:
        class_name = class_id.replace("[]", "").lower()

        if class_name not in class_mapping:
            class_mapping[class_name] = class_index
            class_index += 1

        class_folder = os.path.join(UPLOAD_FOLDER_PATH, f"{class_name}")
        os.makedirs(class_folder, exist_ok=True)

        for file in request.files.getlist(class_id):
            if not file.filename:
                continue

            if file_extension_validation(file.filename):
                filepath = os.path.join(class_folder, file.filename)
                file.save(filepath)

    create_zip('sounds', 'sound_dataset')
    shutil.rmtree('sounds/sound_dataset')

    json_data = json.dumps(class_mapping, ensure_ascii=False, indent=2)
    json_filename = os.path.join(UTILS_PATH, 'class_to_idx.json')
    with open(json_filename, "w", encoding='utf-8') as json_file:
        json_file.write(json_data)

    subprocess.run(['datasphere', 'project', 'job', 'execute',
                    '-p', PROJECT_ID, '-c', CONFIG_YAML_FILE],
                   cwd='utils/sound_classification',
                   capture_output=False,
                   text=True)

    return redirect(url_for('sound_predictor.index'))

@sound_classification_bp.route('/save_sound_hyperparameters', methods=['POST'])
def save_hyperparameters():
    try:
        data = request.get_json()

        hyperparams = {
            "num_epochs": int(data.get('num_epochs', 10)),
            "learning_rate": float(data.get('learning_rate', 0.0001)),
            "batch_size": int(data.get('batch_size', 8)),
            "weight_decay": float(data.get('weight_decay', 0.0001)),
            "target_size": int(data.get('target_size', 50))
        }

        hyperparams_path = os.path.join(UTILS_PATH, 'hyperparams.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)

        return jsonify({
            "success": True,
            "message": "Hyperparameters saved successfully",
            "hyperparameters": hyperparams
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to save hyperparameters: {str(e)}"
        }), 500