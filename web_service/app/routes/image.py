from flask import Blueprint, render_template


image_bp = Blueprint('image', __name__)


@image_bp.route('/image_page', methods = ['GET'])
def image():
    return render_template('index.html')