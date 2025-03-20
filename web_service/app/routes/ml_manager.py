import time
from flask import Blueprint, jsonify, render_template, request, redirect, url_for

from ..fitter import models
from ..models import Dataset
from ..fitter import Fitter

ml_manger_bp = Blueprint('ml_manager', __name__)

@ml_manger_bp.route('/ml_manager', methods=['GET'])
def index():
    datasets = Dataset.query.all()
    return render_template('ml_manager.html', models=models, datasets=datasets, active_page='classic_ml')

@ml_manger_bp.route('/get_model_params')
def get_model_params():
    model_name = request.args.get('model')
    return jsonify(models.get(model_name, {}))

@ml_manger_bp.route('/train', methods=['POST'])
def train_model():
    model_name = request.form['model']
    dataset_id = request.form['dataset']

    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({
            'status': 'error',
            'message': 'Dataset not found'
        }), 400

    params = {}
    # Обрабатываем только параметры, относящиеся к выбранной модели
    for key in request.form:
        if key.startswith('param_'):
            param_name = key.replace('param_', '')
            raw_value = request.form[key]
            param_config = models[model_name].get(param_name)
            # Если значение не передано, используем значение по умолчанию
            if raw_value == "" or raw_value is None:
                params[param_name] = param_config['default']
            else:
                if param_config['type'] == 'int':
                    params[param_name] = int(raw_value)
                elif param_config['type'] == 'float':
                    params[param_name] = float(raw_value)
                else:
                    params[param_name] = raw_value

    fitter = Fitter(model_name, params, dataset)
    fitter.fit()

    return redirect('/tracking')
