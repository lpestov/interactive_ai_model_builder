# image.py
from flask import Blueprint, jsonify, redirect, render_template, request, url_for, flash, Response
from dotenv import load_dotenv
import os
import shutil
import json
import subprocess
import uuid

from app.exceptions.folderNotFoundError import FolderNotFoundError

load_dotenv()

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER_PATH = 'images/classification_dataset/train'
UTILS_PATH = 'utils/image_classification'
YC_TOKEN = os.environ.get('YC_TOKEN')
PROJECT_ID = os.environ.get('PROJECT_ID')
CONFIG_YAML_FILE = os.environ.get('CONFIG_YAML_FILE')

# Словарь для хранения активных процессов обучения
# Ключ: job_id, Значение: {'process': Popen_object, 'status': 'running'/'completed'/'failed'}
active_training_processes = {}

# Функция проверки расширения загруженного файла
def file_extension_validation(filename):
    return "." in filename and filename.split(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Функция для создания zip-архива из имеющейся папки
def create_zip(source_folder, zip_name):
    if not os.path.exists(source_folder):
        raise FolderNotFoundError(f"Folder {source_folder} not found")
    shutil.make_archive(os.path.join(UTILS_PATH, zip_name), 'zip', source_folder)

image_bp = Blueprint('image', __name__)

@image_bp.route('/image_page', methods=['GET'])
def index():
    return render_template('upload_image.html', active_page='image')

@image_bp.route('/upload_images', methods=['POST'])
def upload_images():
    os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)

    if not request.files:
        return jsonify({"success": False, "message": "No files were uploaded."}), 400

    class_file_count = {}
    for class_id in request.files:
        class_name = class_id.replace("[]", "").lower()
        files = request.files.getlist(class_id)
        valid_files = [f for f in files if f.filename and file_extension_validation(f.filename)]
        if class_name not in class_file_count:
            class_file_count[class_name] = 0
        class_file_count[class_name] += len(valid_files)

    invalid_classes = {name: count for name, count in class_file_count.items() if count < 15}
    if invalid_classes:
        error_message = "There are not enough images for the following classes: "
        error_message += ", ".join([f"{name} ({count}/15)" for name, count in invalid_classes.items()])
        return jsonify({"success": False, "message": error_message}), 400

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

    try:
        create_zip('images', 'classification_dataset')
    except FolderNotFoundError as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        if os.path.exists('images/classification_dataset'):
            shutil.rmtree('images/classification_dataset')


    json_data = json.dumps(class_mapping, ensure_ascii=False, indent=2)
    json_filename = os.path.join(UTILS_PATH, 'class_to_idx.json')
    with open(json_filename, "w", encoding='utf-8') as json_file:
        json_file.write(json_data)

    job_id = str(uuid.uuid4())
    command = ['datasphere', 'project', 'job', 'execute',
               '-p', PROJECT_ID, '-c', CONFIG_YAML_FILE]
    process_cwd = UTILS_PATH

    try:
        process = subprocess.Popen(
            command,
            cwd=process_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        active_training_processes[job_id] = {'process': process, 'status': 'running'}
        return jsonify({"success": True, "message": "Training process started.", "job_id": job_id})
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to start training process: {str(e)}"}), 500

def generate_log_stream_for_job(job_id, redirect_url_on_success, processes_dict):
    if job_id not in processes_dict or not processes_dict[job_id].get('process'):
        yield f"data: Error: Job ID {job_id} not found or process not started.\n\n"
        yield f"data: EVENT:STREAM_ENDED\n\n"
        return

    process_info = processes_dict[job_id]
    process = process_info['process']

    yield f"data: Starting training log for job {job_id}...\n\n"

    try:
        for line in iter(process.stdout.readline, ''):
            processed_line = line.rstrip('\n\r')
            yield f"data: {processed_line}\n\n"

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            process_info['status'] = 'completed'
            yield f"data: EVENT:TRAINING_SUCCESS\n\n"
            yield f"data: Training completed successfully. You will be redirected shortly.\n\n"
            yield f"data: REDIRECT:{redirect_url_on_success}\n\n"
        else:
            process_info['status'] = 'failed'
            yield f"data: EVENT:TRAINING_FAILED\n\n"
            yield f"data: Training failed with exit code {return_code}.\n\n"
    except Exception as e:
        process_info['status'] = 'failed'
        yield f"data: EVENT:TRAINING_ERROR\n\n"
        yield f"data: An error occurred while streaming logs: {str(e)}\n\n"
    finally:
        if job_id in processes_dict:
            processes_dict[job_id]['process'] = None
        yield f"data: EVENT:STREAM_ENDED\n\n"

@image_bp.route('/stream_image_logs/<job_id>')
def stream_image_logs(job_id):
    redirect_url = url_for('image_predictor.index')
    return Response(generate_log_stream_for_job(job_id, redirect_url, active_training_processes), mimetype='text/event-stream')

@image_bp.route('/save_hyperparameters', methods=['POST'])
def save_hyperparameters():
    try:
        data = request.get_json()
        hyperparams = {
            "num_epochs": int(data.get('num_epochs', 10)),
            "learning_rate": float(data.get('learning_rate', 0.0001)),
            "batch_size": int(data.get('batch_size', 8)),
            "weight_decay": float(data.get('weight_decay', 0.0001))
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