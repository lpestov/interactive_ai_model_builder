from flask import Blueprint, jsonify, render_template, request, redirect, url_for, flash

from ..automlFitter import AutoMLFitter
from ..models import Dataset

# Конфигурация методов оптимизации гиперпараметров (без random_state и verbose)
hpo_methods = {
    "Bayesian": {
        "name": "Bayesian Optimization",
        "init_points": {"type": "int", "default": 10, "label": "Init Points"},
        "n_iter": {"type": "int", "default": 50, "label": "Iterations"},
        "acq": {"type": "select", "default": "ucb", "label": "Acquisition Function", "options": ["ucb", "ei"]},
        "kappa": {"type": "float", "default": 2.576, "label": "Kappa"},
        "xi": {"type": "float", "default": 0.0, "label": "Xi"},
        "n_candidates": {"type": "int", "default": 500, "label": "Candidates"}
    },
    "Evolutionary": {
        "name": "Evolutionary Strategy",
        "population_size": {"type": "int", "default": 20, "label": "Population Size"},
        "mutation_variance": {"type": "float", "default": 0.1, "label": "Mutation Variance"},
        "generations": {"type": "int", "default": 10, "label": "Generations"},
        "mutation_rate": {"type": "float", "default": 0.25, "label": "Mutation Rate"},
        "mutation_ratio": {"type": "float", "default": 0.75, "label": "Mutation Ratio"},
        "elite_ratio": {"type": "float", "default": 0.2, "label": "Elite Ratio"}
    }
}

# Дефолтные модели (f из sklearn) и их пространство поиска параметров (param_space)
default_models = {
    "RandomForestClassifier": {
        "param_space": [
            {"name": "n_estimators", "type": "integer", "default_low": 50, "default_high": 200},
            {"name": "max_depth", "type": "integer", "default_low": 3, "default_high": 20},
            {"name": "min_samples_split", "type": "integer", "default_low": 2, "default_high": 10},
            {"name": "min_samples_leaf", "type": "integer", "default_low": 1, "default_high": 4},
            {"name": "min_impurity_decrease", "type": "float", "default_low": 0.0, "default_high": 1.0}
        ]
    },
    "LogisticRegression": {
        "param_space": [
            {"name": "C", "type": "float", "default_low": 0.01, "default_high": 10.0},
            {"name": "max_iter", "type": "integer", "default_low": 50, "default_high": 200}
        ]
    },
    "SVC": {
        "param_space": [
            {"name": "C", "type": "float", "default_low": 0.1, "default_high": 10.0},
            {"name": "gamma", "type": "float", "default_low": 0.001, "default_high": 1.0}
        ]
    }
}
sklearn_models = list(default_models.keys())

auto_ml_bp = Blueprint('auto_ml', __name__)


@auto_ml_bp.route('/auto_ml', methods=['GET'])
def index():
    datasets = Dataset.query.all()
    return render_template("auto_ml.html",
                           datasets=datasets,
                           hpo_methods=hpo_methods,
                           sklearn_models=sklearn_models,
                           active_page="auto_ml")


@auto_ml_bp.route('/get_hpo_params')
def get_hpo_params():
    method = request.args.get('method')
    params = hpo_methods.get(method, {})
    return jsonify(params)


@auto_ml_bp.route('/get_model_param_space')
def get_model_param_space():
    model = request.args.get('model')
    model_config = default_models.get(model, {})
    param_space = model_config.get("param_space", [])
    param_dict = {param["name"]: param for param in param_space}
    return jsonify(param_dict)


@auto_ml_bp.route('/auto_ml/train', methods=['POST'])
def train_model():
    dataset_id = request.form['dataset']
    hpo_method = request.form['hpo_method']
    model_name = request.form['model']

    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'status': 'error', 'message': 'Dataset not found'}), 400

    # Собираем параметры метода HPO (без изменений)
    hpo_params = {}
    for key in request.form:
        if key.startswith('hpo_param_') and key != 'hpo_param_name':
            param_name = key.replace('hpo_param_', '')
            value = request.form[key]
            config = hpo_methods[hpo_method].get(param_name)
            if value == "" or value is None:
                hpo_params[param_name] = config['default']
            else:
                if config['type'] == 'int':
                    hpo_params[param_name] = int(value)
                elif config['type'] == 'float':
                    hpo_params[param_name] = float(value)
                else:
                    hpo_params[param_name] = value

    # Собираем пространство поиска параметров для выбранной модели
    model_param_space = {}
    for key in request.form:
        if key.startswith('model_param_'):
            suffix = key[len('model_param_'):]
            parts = suffix.rsplit('_', 1)
            if len(parts) != 2:
                continue

            param_name, bound_type = parts[0], parts[1]
            if bound_type not in ('low', 'high'):
                continue

            model_config = default_models.get(model_name, {})
            param_space = model_config.get("param_space", [])
            param_config = next((p for p in param_space if p["name"] == param_name), None)
            if not param_config:
                continue

            try:
                value = request.form[key]
                if param_config["type"] == "float":
                    value = float(value)
                else:
                    value = int(value)

                if param_name not in model_param_space:
                    model_param_space[param_name] = {}

                model_param_space[param_name][bound_type] = value
            except (ValueError, KeyError) as e:
                flash(f"Ошибка в параметре {param_name}: {str(e)}", "error")
                return redirect(url_for("auto_ml.index"))

    # Конвертируем в требуемый формат
    param_space_list = []
    for param_name, bounds in model_param_space.items():
        model_config = default_models.get(model_name, {})
        param_space = model_config.get("param_space", [])
        param_config = next((p for p in param_space if p["name"] == param_name), None)
        if not param_config:
            continue

        param_space_list.append({
            "name": param_name,
            "type": param_config["type"],
            "low": bounds["low"],
            "high": bounds["high"]
        })

    # Запускаем обучение через AutoMLFitter
    fitter = AutoMLFitter(
        hpo_method,
        hpo_params,
        model_name,
        model_param_space=param_space_list,
        dataset=dataset
    )
    fitter.fit()

    flash("Обучение модели AutoML запущено! Результаты доступны в разделе Tracking.", "success")
    return redirect(url_for("tracking.index"))