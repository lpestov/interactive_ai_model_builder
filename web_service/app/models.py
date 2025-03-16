from datetime import datetime

from .extentions import db

class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String, nullable=False)
    file_path = db.Column(db.String, nullable=False)

    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path

    def __repr__(self):
        return {"id" : self.id, "file_name" : self.file_name, "file_path" : self.file_path}
