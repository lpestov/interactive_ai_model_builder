from flask import Blueprint, jsonify, render_template, request
import os


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
UPLOAD_FOLDER = "images_dataset"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# функция проверки расширения загруженного файла
def file_extension_validation(filename):
    return "." in filename and filename.split(".", 1)[1].lower() in ALLOWED_EXTENSIONS


image_bp = Blueprint('image', __name__)


@image_bp.route('/image_page', methods = ['GET'])
def image():
    return render_template('upload_image.html')

@image_bp.route('/upload_images', methods = ['POST'])
def upload_images():
    files = request.files
    if not files:
        return jsonify({"message": "Нет загруженных файлов"}), 400
    
    saved_files = {}
    
    for key in files:
        class_id = key.replace("file", "").replace("[]", "")
        class_folder = os.path.join(UPLOAD_FOLDER, f"class_{class_id}")
        os.makedirs(class_folder, exist_ok=True)
        
        saved_files[class_id] = []
        
        for file in request.files.getlist(key):
            if not file.filename:
                continue
            
            if file_extension_validation(file.filename):
                filepath = os.path.join(class_folder, file.filename)
                file.save(filepath)
                
                saved_files[class_id].append(file.filename)
            
    return jsonify({
        "message": "Файлы успешно загружены",
        "saved_files": saved_files
    })