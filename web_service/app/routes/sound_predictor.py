from flask import Blueprint, flash, render_template, request, redirect, url_for, send_file
import subprocess
import os
import re
import shutil

INFERENCE_SCRIPT_PATH = 'utils/sound_classification/inference.py'
UPLOAD_FOLDER_PATH = 'sounds'
STATIC_FOLDER = 'app/static'  # папка для статических файлов
MODEL_PATH = 'utils/sound_classification/trained_sound_model.pt'  # путь к обученной модели

sound_predictor_bp = Blueprint('sound_predictor', __name__)

def sound_extension_validation(filename):
    allowed_extensions = {'wav', 'mp3', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@sound_predictor_bp.route('/sound_prediction', methods=['GET'])
def index():
    return render_template('sound_prediction.html')

@sound_predictor_bp.route('/download_model', methods=['GET'])
def download_model():
    print(f"Attempting to download model from: {MODEL_PATH}")
    print(f"File exists check: {os.path.exists(MODEL_PATH)}")

    if os.path.exists(MODEL_PATH):
        try:
            model_absolute_path = os.path.abspath(MODEL_PATH)
            return send_file(model_absolute_path, 
                           as_attachment=True,
                           download_name="trained_sound_model.pt")
        except Exception as e:
            error_msg = f'Error downloading model: {str(e)}'
            print(error_msg)
            flash(error_msg)
            return redirect(url_for('sound_predictor.index'))
    else:
        error_msg = f'Model not found: {MODEL_PATH}'
        print(error_msg)
        flash(error_msg)
        return redirect(url_for('sound_predictor.index'))

@sound_predictor_bp.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(UPLOAD_FOLDER_PATH):
        os.makedirs(UPLOAD_FOLDER_PATH)

    if 'sound' not in request.files:
        flash('No file selected')
        return redirect(url_for('sound_predictor.index'))

    file = request.files['sound']

    if not sound_extension_validation(file.filename):
        flash('Invalid file format')
        return redirect(url_for('sound_predictor.index'))

    file_path = os.path.join(UPLOAD_FOLDER_PATH, file.filename)
    file.save(file_path)

    try:
        result = subprocess.run(
            ['python', INFERENCE_SCRIPT_PATH, file_path],
            capture_output=True, 
            text=True
        )
        
        print(result.returncode)
        print(result.stderr)
        print(result.stdout)

        if result.returncode != 0:
            flash(f"Prediction error: {result.stderr}")
            return redirect(url_for('sound_predictor.index'))

        # Получаем данные из inference-скрипта
        output = result.stdout
        match = re.search(r'Plot saved to: (.*\.png)', output)
        if not match:
            flash('Не удалось найти путь к изображению с диаграммой классов')
            return redirect(url_for('sound_predictor.index'))
        
        # Получаем путь к .png файлу
        spectrogram_path = match.group(1)
        spectrogram_filename = os.path.basename(spectrogram_path)
        
        print(spectrogram_path)
        
        if not os.path.exists(STATIC_FOLDER):
            os.makedirs(STATIC_FOLDER)
        
        # Перемещаем изображение в папку static
        static_spectrogram_path = os.path.join(STATIC_FOLDER, spectrogram_filename)
        shutil.move(spectrogram_path, static_spectrogram_path)
        
        # Формируем URL для изображения (относительный путь от папки static)
        spectrogram_url = spectrogram_filename
        
        return render_template('sound_prediction_result.html', spectrogram_url=spectrogram_url)

    except Exception as e:
        flash(f"Prediction failed: {str(e)}")
        return redirect(url_for('sound_predictor.index'))