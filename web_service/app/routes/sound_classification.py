from flask import Blueprint, render_template


sound_classification_bp = Blueprint('sound_classification', __name__)


@sound_classification_bp.route('/sound_classification')
def index():
    return render_template('sound_classification.html', active_page = "sound")

