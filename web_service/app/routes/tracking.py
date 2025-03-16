import mlflow
from flask import Blueprint, render_template, send_file, abort
from mlflow.tracking import MlflowClient

tracking_bp = Blueprint('tracking', __name__)
client = MlflowClient()


@tracking_bp.route('/tracking', methods=['GET'])
def index():
    runs = mlflow.search_runs()

    experiments = []
    for _, run in runs.iterrows():
        experiments.append({
            'run_id': run.run_id,
            'model_name': run.get('params.model_name', 'Unknown Model'),
            'accuracy': run.get('metrics.accuracy', 0),
            'training_time': run.get('metrics.training_time', 0),
            'params': {k.replace('params.', ''): v
                       for k, v in run.items()
                       if k.startswith('params.') and k != 'params.model_name'},
            'model_uri': f"runs:/{run.run_id}/model"
        })

    return render_template("tracking.html", experiments=experiments, active_page='tracking')

@tracking_bp.route('/download/<run_id>')
def download_model(run_id):
    try:
        path = client.download_artifacts(run_id, "model/model.pkl")
        return send_file(path, as_attachment=True)
    except Exception as e:
        return str(e), 404