from flask import Flask
from .extentions import db

from .routes.main import main_bp
from .routes.image import image_bp
from .routes.ml import ml_bp
from .routes.dataset_manager import dataset_manager_bp
from .routes.table_processor import table_processor_bp
from .routes.tracking import tracking_bp


def create_app():
    # Инициализация приложения
    app = Flask(__name__, template_folder='templates')

    app.secret_key = 'kak i zaebalsya'

    # Инициализация базы данных
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
    db.init_app(app)
    with app.app_context():
        db.create_all()


    app.register_blueprint(main_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(dataset_manager_bp)
    app.register_blueprint(table_processor_bp)
    app.register_blueprint(tracking_bp)

    return app
