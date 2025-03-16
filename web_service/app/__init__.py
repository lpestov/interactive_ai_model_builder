from flask import Flask
from .extentions import db

from .routes.main import main_bp
from .routes.image import image_bp
from .routes.ml_manager import ml_manger_bp
from .routes.dataset_manager import dataset_manager_bp
from .routes.table_processor import table_processor_bp
from .routes.tracking import tracking_bp
from .routes.image_predictor import image_predictor_bp


def create_app():
    # Инициализация приложения
    app = Flask(__name__, template_folder='templates', static_folder='static')

    app.secret_key = 'kak i zaebalsya'

    # Инициализация базы данных
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
    db.init_app(app)
    with app.app_context():
        db.create_all()

    app.register_blueprint(main_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(ml_manger_bp)
    app.register_blueprint(dataset_manager_bp)
    app.register_blueprint(table_processor_bp)
    app.register_blueprint(tracking_bp)
    app.register_blueprint(image_predictor_bp)

    import pandas as pd
    @app.template_filter('datetime')
    def datetime_filter(value):
        return pd.to_datetime(value).strftime('%Y-%m-%d %H:%M:%S')

    return app
