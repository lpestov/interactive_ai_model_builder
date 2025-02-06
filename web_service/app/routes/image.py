from flask import Blueprint, jsonify, render_template, request
import os
import shutil
import json

from web_service.app.exceptions.folderNotFoundError import FolderNotFoundError


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER_PATH = 'images/classification_dataset/train'
UTILS_PATH = '../models/classification/utils'


# функция проверки расширения загруженного файла
def file_extension_validation(filename):
    return "." in filename and filename.split(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# функция для создания zip-архива из имеющейся папки
def create_zip(source_folder, zip_name):
    if not os.path.exists(source_folder):
        raise FolderNotFoundError(f"Папка {source_folder} не существует")
    
    shutil.make_archive(os.path.join(UTILS_PATH, zip_name), 'zip', source_folder)


image_bp = Blueprint('image', __name__)


@image_bp.route('/image_page', methods = ['GET'])
def image():
    return render_template('upload_image.html')

@image_bp.route('/upload_images', methods = ['POST'])
def upload_images():
    os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)
    
    if not request.files:
        return jsonify({"message": "Нет загруженных файлов"}), 400
    
    # отображение имени класса в его порядковый номер (для создания json-а)
    class_mapping = {}
    class_index = 0
    
    for class_id in request.files:
        # получаем имя класса, введенное пользователем в
        # "чистом" виде в нижнем регистре
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
    
    create_zip('images', 'classification_dataset')
    shutil.rmtree('images')
    
    # создание данных для json-а и создание самого файла .json
    json_data = json.dumps(class_mapping, ensure_ascii=False, indent=2)
    json_filename = os.path.join(UTILS_PATH, 'class_to_idx.json')
    with open(json_filename, "w", encoding='utf-8') as json_file:
        json_file.write(json_data)
            
    return jsonify({
        'message': 'Изображения успешно загружены',
        'class_mapping': class_mapping
    })
    