import os
import uuid

from flask import Blueprint, render_template, redirect, request, flash, url_for, send_file

from ..extentions import db
from ..models import Dataset


dataset_manager_bp = Blueprint('dataset_manager', __name__)

@dataset_manager_bp.route('/dataset_manager', methods=['GET'])
def index():
    dataset_list = Dataset.query.all()
    return render_template('dataset_manager.html', datasets=dataset_list, active_page='table_datasets')


@dataset_manager_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Нет файла для загрузки')
        return redirect(url_for('dataset_manager.index'))

    file = request.files['file']
    if file.filename == '':
        flash('Нет выбранного файла')
        return redirect(url_for('dataset_manager.index'))

    if file and file.filename.endswith('.csv'):
        # Генерируем уникальное имя файла
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        new_file_name = f"{unique_id}{file_extension}"

        if not os.path.exists('datasets'):
            os.makedirs('datasets')

        file_path = os.path.join('datasets', new_file_name)
        file_name = file.filename

        file.save(file_path)

        problem_type = request.form['problem_type']
        new_dataset = Dataset(file_name = file_name, file_path = file_path, problem_type = problem_type)
        db.session.add(new_dataset)
        db.session.commit()
        flash('Файл успешно загружен')
        return redirect(url_for('dataset_manager.index'))
    else:
        flash('Неподдерживаемый формат файла. Пожалуйста, загрузите CSV файл.')
        return redirect(url_for('dataset_manager.index'))


@dataset_manager_bp.route('/delete/<int:dataset_id>', methods=['POST'])
def delete_dataset(dataset_id):
    dataset = Dataset.query.get(dataset_id)
    if dataset:
        os.remove(dataset.file_path)
        db.session.delete(dataset)
        db.session.commit()
        flash('Датасет удален')
    else:
        flash('Датасет не найден')

    return redirect(url_for('dataset_manager.index'))

@dataset_manager_bp.route('/download/<int:dataset_id>', methods=['GET'])
def download_dataset(dataset_id):
    dataset = Dataset.query.get(dataset_id)
    if dataset and os.path.exists(dataset.file_path):
        return send_file("../" + dataset.file_path, as_attachment=True, download_name=dataset.file_name)
    else:
        flash('Файл не найден')
        return redirect(url_for('dataset_manager.index'))