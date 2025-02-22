from datetime import datetime

from .extentions import db


class TrainHistory(db.Model):
    __tablename__ = "train_history"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    dataset = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.JSON)
    train_accuracy = db.Column(db.Float)
    test_accuracy = db.Column(db.Float)
    model_path = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(
        self,
        model_name,
        dataset,
        parameters=None,
        train_accuracy=None,
        test_accuracy=None,
        model_path=None,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.parameters = parameters
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.model_path = model_path


class Dataset(db.Model):
    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String, nullable=False)
    file_path = db.Column(db.String, nullable=False)

    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path

    def __repr__(self):  # Исправлено на __repr__
        return f"<Dataset {self.file_name}, {self.file_path}>"
