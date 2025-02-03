from flask import Blueprint, render_template, jsonify

from ..models import TrainHistory

tracking_bp = Blueprint('tracking', __name__)


@tracking_bp.route('/tracking', methods = ['GET'])
def index():
    history = TrainHistory.query.order_by(TrainHistory.created_at.desc()).all()
    return render_template('tracking.html', history = history)

