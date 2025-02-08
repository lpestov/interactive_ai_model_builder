from flask import Blueprint, flash, render_template, request, redirect, url_for
import subprocess
import os
import re
import shutil

from app.routes.image import file_extension_validation

INFERENCE_SCRIPT_PATH = 'utils/inference.py'
UPLOAD_FOLDER_PATH = 'images'
STATIC_FOLDER = 'app/static'  # папка для статических файлов

image_predictor_bp = Blueprint('image_predictor', __name__)

@image_predictor_bp.route('/image_prediction', methods=['GET'])
def index():
    return render_template('image_prediction.html')

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
            flash('Не удалось найти путь к изображению с диаграммой классов')
            return redirect(url_for('image_predictor.index'))

        # Получаем путь к .png файлу
        image_path = match.group(1)
        image_filename = os.path.basename(image_path)

        if not os.path.exists(STATIC_FOLDER):
            os.makedirs(STATIC_FOLDER)

        # Перемещаем изображение в папку static
        static_image_path = os.path.join(STATIC_FOLDER, image_filename)
        print(f"Перемещение {image_path} в {static_image_path}")
        shutil.move(image_path, static_image_path)

        if os.path.exists(static_image_path):
            print(f"Файл успешно перемещен в {static_image_path}")
        else:
            print(f"Ошибка: Файл не найден в {static_image_path}")

        # Формируем URL для изображения (относительный путь от папки static)
        image_url = os.path.join(image_filename)

        print(f"Путь к изображению: {static_image_path}")
        print(f"URL изображения: {image_url}")

        return render_template('prediction_result.html', image_url=image_url)

    except Exception as e:
        flash(f"Ошибка при попытке предсказания: {str(e)}")
        return redirect(url_for('image_predictor.index'))