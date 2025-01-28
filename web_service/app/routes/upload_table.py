from flask import Blueprint, render_template


upload_table_bp = Blueprint('upload_table', __name__)


@upload_table_bp.route('/upload_table_page', methods = ['GET'])
def upload_table():
    return render_template('upload_table.html')