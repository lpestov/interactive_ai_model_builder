from flask import Blueprint, request, render_template

import pandas as pd


table_processing_bp = Blueprint('table_processing', __name__)


@table_processing_bp.route('/table_processing_page', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'Файл не найден'

    file = request.files['file']

    if file.filename == '':
        return 'Выберите файл'

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        return render_template('table_processing.html', tables=[df.to_html(classes='data', header="true", index=False)])

    return 'Файл должен быть в формате CSV'