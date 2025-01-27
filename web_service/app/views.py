from flask import render_template, request, jsonify, redirect, url_for
import pandas as pd
import time

from app import app

models = {
        'RandomForest': {
            'n_estimators': {'type': 'int', 'default': 100, 'description': 'Количество деревьев в лесу'},
            'max_depth': {'type': 'int', 'default': None, 'description': 'Максимальная глубина дерева'},
            'batch_size': {'type': 'int', 'default': 16, 'description': 'Размер батча'}
        },
        'SVM': {
            'C': {'type': 'float', 'default': 1.0, 'description': 'Регуляризационный параметр'},
            'gamma': {'type': 'float', 'default': 'scale', 'description': 'Параметр ядра'},
            'batch_size': {'type': 'int', 'default': 16, 'description': 'Размер батча'}
        }
}

# Список для хранения результатов
tracking_storage = {
    'ongoing': [],
    'completed': []
}


@app.route('/')
def index():
    return render_template('index.html')


# Функция для симуляции обучения модели
def train(model_name, params):
    time.sleep(5)  # Симуляция времени обучения
    # Завершение обучения модели
    result = {
        'model_name': model_name,
        'training_time': "5 seconds",
        'parameters': params,
        'model_path': f"/models/{model_name}.h5"
    }
    tracking_storage['completed'].append(result)

@app.route('/ml_page', methods = ['GET'])
def ml():
    return render_template('ml.html', models=models)

@app.route('/params/<model_name>', methods=['GET'])
def get_params(model_name):
    return jsonify(models.get(model_name, {}))

@app.route('/train', methods=['POST'])
def train_model():
    model_name = request.form['model']
    model_params = {key: value for key, value in request.form.items() if key != 'model'}

    # Обучение модели (здесь должен быть ваш код)
    print(f'Обучаем модель: {model_name} с параметрами: {model_params}')
    train(model_name, model_params, )

    return redirect('tracking_page')









@app.route('/upload_table_page', methods = ['GET'])
def upload_table():
    return render_template('upload_table.html')

@app.route('/table_processing_page', methods=['POST'])
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



@app.route('/tracking_page', methods = ['GET'])
def tracking():
    return render_template('tracking.html')

@app.route('/get_results', methods=['GET'])
def get_results():
    return jsonify(tracking_storage)







@app.route('/image_page', methods = ['GET'])
def image():
    return render_template('index.html')