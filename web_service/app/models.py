from app import db

class ModelTraining(db.Model):
    __tablename__ = 'models'

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    training_date = db.Column(db.DateTime, nullable=False)
    epoch = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    loss = db.Column(db.Float, nullable=False)
    model_file_path = db.Column(db.String(200), nullable=False)

    def __init__(self, model_name, training_date, epoch, accuracy, loss, model_file_path):
        self.model_name = model_name
        self.training_date = training_date
        self.epoch = epoch
        self.accuracy = accuracy
        self.loss = loss
        self.model_file_path = model_file_path

    def save_model(self, model, model_dir='models'):
        """Сохранить модель в файл формата .pt"""
        model_file_path = ''
        self.model_file_path = model_file_path

    @classmethod
    def get_all_models(cls):
        """Получить все записи о моделях"""
        return cls.query.all()