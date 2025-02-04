import os
import uuid

from flask import Blueprint, render_template, redirect, request, flash, url_for

from ..extentions import db
from ..models import Dataset

dataset_manager_bp = Blueprint('dataset_manager', __name__)



@dataset_manager_bp.route('/dataset_manager', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pass
    dataset_list = Dataset.query.all()
    return render_template('dataset_manager.html', datasets=dataset_list)


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

        # Сохраняем путь к загруженному датасету в БД
        new_dataset = Dataset(file_name = file_name, file_path = file_path)
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
        # Удаляем файл из файловой системы
        os.remove(dataset.file_path)

        # Удаляем запись из базы данных
        db.session.delete(dataset)
        db.session.commit()
        flash('Датасет удален')
    else:
        flash('Датасет не найден')

    return redirect(url_for('dataset_manager.index'))