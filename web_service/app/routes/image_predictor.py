from flask import Blueprint, jsonify, render_template


image_predictor_bp = Blueprint('image_predictor', __name__)


@image_predictor_bp.route('/image_prediction', methods = ['GET'])
def index():
    return render_template('image_prediction.html')

@image_predictor_bp.route('/predict_image_class', methods = ['POST'])
def predict():
    return jsonify({'status': 'good'}), 200