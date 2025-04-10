import time
import csv
import threading
from flask import Blueprint, jsonify, render_template, request, redirect, url_for
from ..fitter import classification_models, regression_models, Fitter
from ..models import Dataset

ml_manager_bp = Blueprint('ml_manager', __name__)

@ml_manager_bp.route('/ml_manager', methods=['GET'])
def index():
    datasets = Dataset.query.all()

    # Вычисляем, доступны ли датасеты для каждого типа задач
    classification_available = any(d.problem_type == 'classification' for d in datasets)
    regression_available = any(d.problem_type == 'regression' for d in datasets)

    classification_metrics = [
        {'value': 'accuracy', 'label': 'Accuracy'},
        {'value': 'f1', 'label': 'F1 Score'}
    ]
    regression_metrics = [
        {'value': 'r2', 'label': 'R2 Score'},
        {'value': 'mse', 'label': 'Mean Squared Error'}
    ]

    datasets_json = [{
        'id': d.id,
        'file_name': d.file_name,
        'problem_type': d.problem_type,
        'process_status': d.process_status
    } for d in datasets]

    return render_template(
        'ml_manager.html',
        classification_models=classification_models,
        regression_models=regression_models,
        datasets_json=datasets_json,
        classification_metrics=classification_metrics,
        regression_metrics=regression_metrics,
        classification_available=classification_available,
        regression_available=regression_available,
        active_page='classic_ml'
    )

@ml_manager_bp.route('/get_model_params')
def get_model_params():
    model_name = request.args.get('model')
    if model_name in classification_models:
        params = classification_models[model_name]
    elif model_name in regression_models:
        params = regression_models[model_name]
    else:
        params = {}
    return jsonify(params)

@ml_manager_bp.route('/get_target_columns')
def get_target_columns():
    dataset_id = request.args.get('dataset_id')
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify([])
    try:
        with open(dataset.file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            columns = next(reader)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify(columns)

@ml_manager_bp.route('/train', methods=['POST'])
def train_model():
    dataset_id = request.form.get('dataset')
    task_type = request.form.get('task_type')
    target_column = request.form.get('target_column')

    if not dataset_id or not task_type or not target_column:
        return jsonify({'error': 'Не выбраны обязательные параметры: датасет, тип задачи или целевая колонка'}), 400

    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'error': 'Выбранный датасет не найден'}), 400

    model_name = request.form['model']
    scoring = request.form['scoring']
    split_ratio = float(request.form.get('split_ratio', 70))

    params = {}
    for key in request.form:
        if key.startswith('param_'):
            param_name = key.replace('param_', '')
            model_dict = classification_models if task_type == 'classification' else regression_models
            param_config = model_dict.get(model_name, {}).get(param_name, {})
            raw_value = request.form[key]
            if raw_value == "" or raw_value is None:
                params[param_name] = param_config.get('default')
            else:
                if param_config.get('type') == 'int':
                    params[param_name] = int(raw_value)
                elif param_config.get('type') == 'float':
                    params[param_name] = float(raw_value)
                else:
                    params[param_name] = raw_value

    def async_train():
        try:
            fitter = Fitter(
                task_type=task_type,
                model_name=model_name,
                params=params,
                dataset=dataset,
                split_ratio=split_ratio,
                target_column=target_column,
                scoring=scoring
            )
            fitter.fit()
        except Exception as e:
            print(f"Training failed: {str(e)}")

    train_thread = threading.Thread(target=async_train)
    train_thread.start()
    time.sleep(0.2)

    return redirect(url_for('tracking.index'))
