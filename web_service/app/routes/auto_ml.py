import threading
import time
import csv
from flask import Blueprint, jsonify, render_template, request, redirect, url_for, flash
from ..automlFitter import AutoMLFitter
from ..models import Dataset

hpo_methods = {
    "Bayesian": {
        "name": "Bayesian Optimization",
        "init_points": {
            "type": "int",
            "default": 10,
            "label": "Init Points",
            "description": "Number of initial random evaluations before starting the Bayesian optimization."
        },
        "n_iter": {
            "type": "int",
            "default": 50,
            "label": "Iterations",
            "description": "Number of iterations for the optimization process."
        },
        "acq": {
            "type": "select",
            "default": "ucb",
            "label": "Acquisition Function",
            "options": ["ucb", "ei"],
            "description": "Acquisition function to determine the next evaluation point."
        },
        "kappa": {
            "type": "float",
            "default": 2.58,
            "label": "Kappa",
            "description": "Controls the exploration/exploitation trade-off in UCB acquisition."
        },
        "xi": {
            "type": "float",
            "default": 0.0,
            "label": "Xi",
            "description": "Exploration parameter for Expected Improvement (EI) acquisition."
        },
        "n_candidates": {
            "type": "int",
            "default": 500,
            "label": "Candidates",
            "description": "Number of candidate points to evaluate."
        }
    },
    "Evolutionary": {
        "name": "Evolutionary Strategy",
        "population_size": {
            "type": "int",
            "default": 20,
            "label": "Population Size",
            "description": "Number of individuals in the population."
        },
        "mutation_variance": {
            "type": "float",
            "default": 0.1,
            "label": "Mutation Variance",
            "description": "Variance of the mutation noise applied to individuals."
        },
        "generations": {
            "type": "int",
            "default": 10,
            "label": "Generations",
            "description": "Number of generations to run the evolutionary strategy."
        },
        "mutation_rate": {
            "type": "float",
            "default": 0.25,
            "label": "Mutation Rate",
            "description": "Probability of mutation per individual."
        },
        "mutation_ratio": {
            "type": "float",
            "default": 0.75,
            "label": "Mutation Ratio",
            "description": "Proportion of the population that will undergo mutation."
        },
        "elite_ratio": {
            "type": "float",
            "default": 0.2,
            "label": "Elite Ratio",
            "description": "Proportion of top individuals carried over unchanged to the next generation."
        }
    }
}

classification_models = {
    "RandomForestClassifier": {
        "param_space": [
            {
                "name": "n_estimators",
                "type": "integer",
                "default_low": 50,
                "default_high": 200,
                "description": "Number of trees in the forest."
            },
            {
                "name": "max_depth",
                "type": "integer",
                "default_low": 3,
                "default_high": 20,
                "description": "Maximum depth of each tree."
            },
            {
                "name": "min_samples_split",
                "type": "integer",
                "default_low": 2,
                "default_high": 10,
                "description": "Minimum number of samples required to split an internal node."
            },
            {
                "name": "min_samples_leaf",
                "type": "integer",
                "default_low": 1,
                "default_high": 4,
                "description": "Minimum number of samples required to be at a leaf node."
            },
            {
                "name": "min_impurity_decrease",
                "type": "float",
                "default_low": 0.0,
                "default_high": 1.0,
                "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value."
            }
        ]
    },
    "LogisticRegression": {
        "param_space": [
            {
                "name": "C",
                "type": "float",
                "default_low": 0.01,
                "default_high": 10.0,
                "description": "Inverse of regularization strength; smaller values specify stronger regularization."
            },
            {
                "name": "max_iter",
                "type": "integer",
                "default_low": 50,
                "default_high": 200,
                "description": "Maximum number of iterations for the solver."
            }
        ]
    },
    "SVC": {
        "param_space": [
            {
                "name": "C",
                "type": "float",
                "default_low": 0.1,
                "default_high": 10.0,
                "description": "Regularization parameter."
            },
            {
                "name": "gamma",
                "type": "float",
                "default_low": 0.01,
                "default_high": 1.0,
                "description": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'."
            }
        ]
    },
    "GradientBoostingClassifier": {
        "param_space": [
            {
                "name": "n_estimators",
                "type": "integer",
                "default_low": 50,
                "default_high": 200,
                "description": "Number of boosting stages."
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default_low": 0.01,
                "default_high": 0.3,
                "description": "Shrinks the contribution of each tree."
            }
        ]
    }
}

regression_models = {
    "RandomForestRegressor": {
        "param_space": [
            {
                "name": "n_estimators",
                "type": "integer",
                "default_low": 50,
                "default_high": 200,
                "description": "Number of trees in the forest."
            },
            {
                "name": "max_depth",
                "type": "integer",
                "default_low": 3,
                "default_high": 20,
                "description": "Maximum depth of the trees."
            }
        ]
    },
    "SVR": {
        "param_space": [
            {
                "name": "C",
                "type": "float",
                "default_low": 0.1,
                "default_high": 10.0,
                "description": "Regularization parameter."
            },
            {
                "name": "epsilon",
                "type": "float",
                "default_low": 0.01,
                "default_high": 0.2,
                "description": "Epsilon in the epsilon-SVR model."
            }
        ]
    },
    "GradientBoostingRegressor": {
        "param_space": [
            {
                "name": "n_estimators",
                "type": "integer",
                "default_low": 50,
                "default_high": 200,
                "description": "Number of boosting stages."
            },
            {
                "name": "max_depth",
                "type": "integer",
                "default_low": 3,
                "default_high": 10,
                "description": "Maximum depth of the trees."
            }
        ]
    }
}

classification_metrics = [
    {'value': 'accuracy', 'label': 'Accuracy'},
    {'value': 'f1', 'label': 'F1 Score'}
]
regression_metrics = [
    {'value': 'r2', 'label': 'R2 Score'},
    {'value': 'mse', 'label': 'Mean Squared Error'}
]

auto_ml_bp = Blueprint('auto_ml', __name__)

@auto_ml_bp.route('/auto_ml', methods=['GET'])
def index():
    datasets = Dataset.query.all()
    datasets_json = [{
        "id": ds.id,
        "file_name": ds.file_name,
        "problem_type": ds.problem_type,
        "process_status": ds.process_status
    } for ds in datasets]

    classification_available = any(ds.problem_type == 'classification' and ds.process_status for ds in datasets)
    regression_available = any(ds.problem_type == 'regression' and ds.process_status for ds in datasets)

    return render_template("auto_ml.html",
                           datasets=datasets,
                           datasets_json=datasets_json,
                           hpo_methods=hpo_methods,
                           classification_models=classification_models,
                           regression_models=regression_models,
                           classification_metrics=classification_metrics,
                           regression_metrics=regression_metrics,
                           active_page="auto_ml",
                           classification_available=classification_available,
                           regression_available=regression_available)


@auto_ml_bp.route('/get_hpo_params')
def get_hpo_params():
    method = request.args.get('method')
    params = hpo_methods.get(method, {})
    return jsonify(params)

@auto_ml_bp.route('/get_model_param_space')
def get_model_param_space():
    model = request.args.get('model')
    task_type = request.args.get('task_type')
    if task_type == 'classification':
        model_config = classification_models.get(model, {})
    else:
        model_config = regression_models.get(model, {})
    param_space = model_config.get("param_space", [])
    param_dict = {param["name"]: param for param in param_space}
    return jsonify(param_dict)

@auto_ml_bp.route('/get_target_columns')
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

@auto_ml_bp.route('/auto_ml/train', methods=['POST'])
def train_model():
    dataset_id = request.form.get('dataset')
    task_type = request.form.get('task_type')
    target_column = request.form.get('target_column')
    if not dataset_id or not task_type or not target_column:
        flash("Пожалуйста, выберите обязательные поля: датасет, тип задачи и целевую колонку", "error")
        return redirect(url_for("auto_ml.index"))
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        flash("Выбранный датасет не существует", "error")
        return redirect(url_for("auto_ml.index"))
    hpo_method = request.form['hpo_method']
    model_name = request.form['model']
    split_ratio = request.form['split_ratio']
    scoring_select = request.form['scoring']
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
            if task_type == 'classification':
                model_config = classification_models.get(model_name, {})
            else:
                model_config = regression_models.get(model_name, {})
            param_space = model_config.get("param_space", [])
            param_config = next((p for p in param_space if p["name"] == param_name), None)
            if not param_config:
                continue
            try:
                value = request.form[key]
                value = float(value) if param_config["type"] == "float" else int(value)
                if param_name not in model_param_space:
                    model_param_space[param_name] = {}
                model_param_space[param_name][bound_type] = value
            except (ValueError, KeyError) as e:
                flash(f"Ошибка в параметре {param_name}: {str(e)}", "error")
                return redirect(url_for("auto_ml.index"))
    param_space_list = []
    for param_name, bounds in model_param_space.items():
        if task_type == 'classification':
            model_config = classification_models.get(model_name, {})
        else:
            model_config = regression_models.get(model_name, {})
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
    def async_train():
        try:
            fitter = AutoMLFitter(
                hpo_method,
                hpo_params,
                model_name,
                param_space_list,
                dataset,
                task_type,
                split_ratio,
                target_column,
                scoring_select
            )
            fitter.fit()
        except Exception as e:
            print("AutoMl Training error : ", e)
    train_thread = threading.Thread(target=async_train)
    train_thread.start()
    time.sleep(0.2)
    return redirect(url_for("tracking.index"))
