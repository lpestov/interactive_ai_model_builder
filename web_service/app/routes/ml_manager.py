import time

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from ..fitter import Fitter, models
from ..models import Dataset


ml_manger_bp = Blueprint("ml_manager", __name__)


@ml_manger_bp.route("/ml_manager", methods=["GET"])
def index():
    datasets = Dataset.query.all()
    return render_template("ml_manager.html", models=models, datasets=datasets)


@ml_manger_bp.route("/get_model_params")
def get_model_params():
    model_name = request.args.get("model")
    return jsonify(models.get(model_name, {}))


@ml_manger_bp.route("/train", methods=["POST"])
def train_model():
    try:
        # Получаем данные из формы
        model_name = request.form["model"]
        dataset = request.form["dataset"]

        # Собираем параметры модели
        params = {}
        for key in request.form:
            if key.startswith("param_"):
                param_name = key.replace("param_", "")
                raw_value = request.form[key]

                # Преобразуем тип данных согласно спецификации модели
                param_type = models[model_name][param_name]["type"]
                if param_type == "int" and raw_value:
                    params[param_name] = int(raw_value)
                elif param_type == "float" and raw_value:
                    params[param_name] = float(raw_value)
                else:
                    params[param_name] = raw_value if raw_value else None

        # Запускаем оубчение модели
        fitter = Fitter(model_name, params, dataset)
        fitter.fit()

        return redirect(url_for("tracking.index"))

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
