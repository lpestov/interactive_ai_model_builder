import time
from flask import Blueprint, jsonify, render_template, request, redirect

from app.ml_models import models


ml_bp = Blueprint('ml', __name__)


# Список для хранения результатов
tracking_storage = {
    'ongoing': [],
    'completed': []
}


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


@ml_bp.route('/ml_page', methods = ['GET'])
def ml():
    return render_template('ml.html', models=models)

@ml_bp.route('/params/<model_name>', methods=['GET'])
def get_params(model_name):
    return jsonify(models.get(model_name, {}))

@ml_bp.route('/train', methods=['POST'])
def train_model():
    model_name = request.form['model']
    model_params = {key: value for key, value in request.form.items() if key != 'model'}

    # Обучение модели (здесь должен быть ваш код)
    print(f'Обучаем модель: {model_name} с параметрами: {model_params}')
    train(model_name, model_params, )

    return redirect('tracking_page')