import os
import shutil
import ast
import mlflow
from flask import Blueprint, render_template, send_file, redirect, url_for, request
from mlflow.tracking import MlflowClient
from ..fitter import classification_models, regression_models

tracking_bp = Blueprint('tracking', __name__)
client = MlflowClient()

@tracking_bp.route('/tracking', methods=['GET'])
def index():
    runs_df = mlflow.search_runs()
    classic_experiments = []
    automl_experiments = []
    for _, run in runs_df.iterrows():
        model_name = run.get('params.model_name', 'Unknown Model')
        allowed_params = {}
        if model_name in classification_models:
            allowed_params = classification_models.get(model_name, {})
        elif model_name in regression_models:
            allowed_params = regression_models.get(model_name, {})
        status = run.get('tags.status', 'finished')
        model_type = run.get('tags.model_type', 'classic_ml')
        if model_type == 'auto_ml':
            best_params = {}
            best_params_dict = run.get("params.best_params", "{}")
            if best_params_dict:
                for k, v in ast.literal_eval(best_params_dict.replace("np.float64", "")).items():
                    best_params[k] = v
            automl_exp_data = {
                'run_id': run.run_id,
                'status': status,
                'model_name': model_name,
                'score': run.get('metrics.score', 0),
                'training_time': run.get('metrics.training_time', 0),
                'dataset': run.get('params.dataset', 'N/A'),
                'params': best_params,
                'task_type': run.get('params.task_type', 'N/A'),
                'scoring': run.get('params.scoring', 'N/A'),
                'optimizer': run.get('params.hpo_method', 'N/A')
            }
            automl_experiments.append(automl_exp_data)
        else:
            params = {}
            for k, v in run.items():
                if k.startswith('params.') and k not in ['params.model_name', 'params.dataset']:
                    param_key = k.replace('params.', '')
                    if param_key in allowed_params:
                        params[param_key] = v
            classic_exp_data = {
                'run_id': run.run_id,
                'status': status,
                'model_name': model_name,
                'score': run.get('metrics.score', 0),
                'training_time': run.get('metrics.training_time', 0),
                'dataset': run.get('params.dataset', 'N/A'),
                'params': params,
                'task_type': run.get('params.task_type', 'N/A'),
                'scoring': run.get('params.scoring', 'N/A')
            }
            classic_experiments.append(classic_exp_data)
    return render_template("tracking.html",
                           classic_experiments=classic_experiments,
                           automl_experiments=automl_experiments,
                           active_page='tracking')

@tracking_bp.route('/tracking/<run_id>', methods=['GET'])
def tracking_detail(run_id):
    run = client.get_run(run_id)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        snippet_path = os.path.join(current_dir, '..', 'ML_snippet.py')
        with open(snippet_path, 'r') as f:
            code_snippet = f.read()
    except Exception:
        code_snippet = "Snippet not found in the app directory"
    model_type = run.data.tags.get("model_type", "classic_ml")
    if model_type == "auto_ml":
        return render_template("tracking_automl_detail.html", run=run, code_snippet=code_snippet)
    else:
        return render_template("tracking_detail.html", run=run, code_snippet=code_snippet)

@tracking_bp.route('/download/<run_id>')
def download_model(run_id):
    try:
        path = client.download_artifacts(run_id, "model/model.pkl")
        return send_file(path, as_attachment=True)
    except Exception as e:
        return str(e), 404

@tracking_bp.route('/delete/<run_id>', methods=['POST'])
def delete_run(run_id):
    MLRUNS_PATH = os.path.join(os.path.dirname(__file__), '../../mlruns/0')
    folder_path = os.path.join(MLRUNS_PATH, run_id)
    try:
        client.delete_run(run_id)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
        return redirect(url_for('tracking.index'))
    except Exception as e:
        return str(e), 404
