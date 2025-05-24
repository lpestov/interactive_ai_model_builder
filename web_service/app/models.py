from .extentions import db

class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String, nullable=False)
    file_path = db.Column(db.String, nullable=False)
    problem_type = db.Column(db.String, nullable=False)
    process_status = db.Column(db.Boolean, nullable=False)

    def __init__(self, file_name, file_path, problem_type):
        self.file_name = file_name
        self.file_path = file_path
        self.problem_type = problem_type
        if self.check_dataset() != True:
            self.process_status = False


    def check_dataset(self):
        script = True
        if script:
            self.process_status = True
            return True

        return False

    def __repr__(self):
        return {"id" : self.id, "file_name" : self.file_name, "file_path" : self.file_path, "problem_type" : self.problem_type, "process_status" : self.process_status}
