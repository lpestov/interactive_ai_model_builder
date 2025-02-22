from flask import Blueprint, abort, render_template, send_file

from ..extentions import db
from ..models import TrainHistory


tracking_bp = Blueprint("tracking", __name__)


@tracking_bp.route("/tracking", methods=["GET"])
def index():
    history = TrainHistory.query.order_by(TrainHistory.created_at.desc()).all()
    return render_template("tracking.html", history=history)


@tracking_bp.route("/download/<int:record_id>")
def download_model(record_id):
    # Получаем запись из базы данных
    record = db.session.query(TrainHistory).get(record_id)
    if record and record.model_path:
        try:
            return send_file(
                "../" + record.model_path,
                as_attachment=True,
                download_name=record.model_name + "/" + record.model_path,
            )
        except FileNotFoundError:
            abort(404)  # Если файл не найден, возвращаем 404
    else:
        return abort(404)  # Если запись не найдена или путь пуст, возвращаем 404
