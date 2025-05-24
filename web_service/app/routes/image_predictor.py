from flask import Blueprint, flash, render_template, request, redirect, url_for, send_file, abort
import subprocess
import os
import re
import shutil
import tempfile
import zipfile


from app.routes.image import file_extension_validation

INFERENCE_SCRIPT_PATH = 'utils/image_classification/inference.py'
UPLOAD_FOLDER_PATH = 'images'
STATIC_FOLDER = 'app/static'
MODEL_PATH = 'utils/image_classification/trained_model_classification.pt'
СLASS_TO_IDX_PATH = 'utils/image_classification/class_to_idx.json'
HYPERPARAMS_PATH = 'utils/image_classification/hyperparams.json'

image_predictor_bp = Blueprint('image_predictor', __name__)


@image_predictor_bp.route('/image_prediction', methods=['GET'])
def index():
    return render_template('image_prediction.html')


@image_predictor_bp.route('/download_model', methods=['GET'])
def download_model():
    print(f"Attempting to download image model package...")

    model_path = MODEL_PATH
    class_to_idx_path = СLASS_TO_IDX_PATH
    hyperparams_path = HYPERPARAMS_PATH

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
        return redirect(url_for('image_predictor.index'))

    try:

        temp_file = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        with zipfile.ZipFile(temp_file.name, 'w') as zipf:
            zipf.write(model_path, arcname='trained_model_classification.pt')
            zipf.write(class_to_idx_path, arcname='class_to_idx.json')
            zipf.write(hyperparams_path, arcname='hyperparameters.json')

        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name="image_classification_package.zip"
        )
    except Exception as e:
        error_msg = f'Ошибка при создании архива: {str(e)}'
        print(error_msg)
        flash(error_msg)
        return redirect(url_for('image_predictor.index'))


@image_predictor_bp.route('/predict_image_class', methods=['POST'])
def predict():
    if not os.path.exists(UPLOAD_FOLDER_PATH):
        os.makedirs(UPLOAD_FOLDER_PATH)

    if 'image' not in request.files:
        flash('Файл не выбран')
        return redirect(url_for('image_predictor.index'))

    file = request.files['image']

    if not file_extension_validation(file.filename):
        flash('Неверное расширение у файла')
        return redirect(url_for('image_predictor.index'))

    file_path = os.path.join(UPLOAD_FOLDER_PATH, file.filename)
    file.save(file_path)

    try:
        result = subprocess.run(
            ['python', INFERENCE_SCRIPT_PATH, file_path],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            flash(f"Ошибка при исполнении скрипта для предсказания: {result.stderr}")
            return redirect(url_for('image_predictor.index'))

        # Получаем данные из inference-скрипта
        output = result.stdout
        match = re.search(r'Plot saved to: (.*\.png)', output)
        if not match:
            flash('Could not find path to class diagram image')
            return redirect(url_for('image_predictor.index'))

        # Получаем путь к .png файлу
        image_path = match.group(1)
        image_filename = os.path.basename(image_path)

        if not os.path.exists(STATIC_FOLDER):
            os.makedirs(STATIC_FOLDER)

        # Перемещаем изображение в папку static
        static_image_path = os.path.join(STATIC_FOLDER, image_filename)
        shutil.move(image_path, static_image_path)

        # Формируем URL для изображения (относительный путь от папки static)
        image_url = os.path.join(image_filename)

        return render_template('image_prediction_result.html', image_url=image_url)

    except Exception as e:
        flash(f"Ошибка при попытке предсказания: {str(e)}")
        return redirect(url_for('image_predictor.index'))
