import os
import shutil

import mlflow
from flask import Blueprint, render_template, send_file, redirect, url_for, request
from mlflow.tracking import MlflowClient
from ..fitter import classification_models, regression_models

tracking_bp = Blueprint('tracking', __name__)
client = MlflowClient()

@tracking_bp.route('/tracking', methods=['GET'])
def index():
    runs = mlflow.search_runs()
    experiments = []
    for _, run in runs.iterrows():
        model_name = run.get('params.model_name', 'Unknown Model')
        allowed_params = {}
        if model_name in classification_models:
            allowed_params = classification_models.get(model_name, {})
        elif model_name in regression_models:
            allowed_params = regression_models.get(model_name, {})
        params = {}
        for k, v in run.items():
            if k.startswith('params.') and k not in ['params.model_name', 'params.dataset']:
                param_key = k.replace('params.', '')
                if param_key in allowed_params:
                    params[param_key] = v
        experiments.append({
            'run_id': run.run_id,
            'model_name': model_name,
            'score': run.get('metrics.score', 0),
            'training_time': run.get('metrics.training_time', 0),
            'dataset': run.get('params.dataset', 'N/A'),
            'params': params,
            'model_uri': f"runs:/{run.run_id}/model",
            'task_type': run.get('params.task_type', 'N/A'),
            'scoring': run.get('params.scoring', 'N/A')
        })
    return render_template("tracking.html", experiments=experiments, active_page='tracking')

@tracking_bp.route('/tracking/<run_id>', methods=['GET'])
def tracking_detail(run_id):
    run = client.get_run(run_id)
    return render_template("tracking_detail.html", run=run)

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
