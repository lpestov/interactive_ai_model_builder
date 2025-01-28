from flask import Blueprint, render_template, jsonify


tracking_bp = Blueprint('tracking', __name__)


@tracking_bp.route('/tracking_page', methods = ['GET'])
def tracking():
    return render_template('tracking.html')

@tracking_bp.route('/get_results', methods=['GET'])
def get_results():
    return jsonify(tracking_storage)