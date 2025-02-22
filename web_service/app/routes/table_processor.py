import os

import pandas as pd
from flask import Blueprint, render_template, request

from ..extentions import db
from ..models import Dataset


table_processor_bp = Blueprint("table_processor", __name__)


@table_processor_bp.route("/process_dataset/<int:dataset_id>", methods=["GET"])
def process_dataset(dataset_id):
    dataset = Dataset.query.get(dataset_id)
    if dataset:
        df = pd.read_csv(dataset.file_path)
        return render_template(
            "table_processing.html",
            tables=[df.to_html(classes="data", header="true", index=False)],
        )

    return "Датасет не найден", 404
