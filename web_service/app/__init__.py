from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

# Инициализация приложения
app = Flask(__name__, template_folder='templates')

# Инициализация базы данных
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
db = SQLAlchemy(app)
with app.app_context():
    if not os.path.exists('sqlite:///blog.db'):
        db.create_all()

from app.routes.main import main_bp
from app.routes.image import image_bp
from app.routes.ml import ml_bp
from app.routes.upload_table import upload_table_bp
from app.routes.table_processing import table_processing_bp
from app.routes.tracking import tracking_bp

app.register_blueprint(main_bp)
app.register_blueprint(image_bp)
app.register_blueprint(ml_bp)
app.register_blueprint(upload_table_bp)
app.register_blueprint(table_processing_bp)
app.register_blueprint(tracking_bp)

from app import routes, models