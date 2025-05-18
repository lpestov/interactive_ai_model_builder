from flask import Blueprint, flash, render_template, request, redirect, url_for, send_file
import subprocess
import os
import re
import shutil

INFERENCE_SCRIPT_PATH = 'utils/sound_classification/inference.py'
UPLOAD_FOLDER_PATH = 'sounds'
STATIC_FOLDER = 'app/static'  # папка для статических файлов
MODEL_PATH = 'utils/sound_classification/trained_model_sound_classification.pt'  # путь к обученной модели

sound_predictor_bp = Blueprint('sound_predictor', __name__)

def sound_extension_validation(filename):
    allowed_extensions = {'wav', 'mp3', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@sound_predictor_bp.route('/sound_prediction', methods=['GET'])
def index():
    return render_template('sound_prediction.html')

@sound_predictor_bp.route('/download_sound_model', methods=['GET'])
def download_model():
    print(f"Attempting to download sound model package...")

    model_path = 'utils/sound_classification/trained_model_sound_classification.pt'
    class_to_idx_path = 'utils/sound_classification/class_to_idx.json'
    hyperparams_path = 'utils/sound_classification/hyperparams.json'

    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append('модель')
    if not os.path.exists(class_to_idx_path):
        missing_files.append('сопоставление классов')
    if not os.path.exists(hyperparams_path):
        missing_files.append('гиперпараметры')

    if missing_files:
        error_msg = f'Файлы не найдены: {", ".join(missing_files)}'
        print(error_msg)
        flash(error_msg)
        return redirect(url_for('sound_predictor.index'))

    try:
        import tempfile
        import zipfile

        temp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        with zipfile.ZipFile(temp_file.name, 'w') as zipf:
            zipf.write(model_path, arcname='trained_model_sound_classification.pt')

            zipf.write(class_to_idx_path, arcname='class_to_idx.json')

            zipf.write(hyperparams_path, arcname='hyperparameters.json')

        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name="sound_classification_package.zip"
        )
    except Exception as e:
        error_msg = f'Ошибка при создании архива: {str(e)}'
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