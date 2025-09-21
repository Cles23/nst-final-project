# app/__init__.py
from flask import Flask
import os

def create_app():
    app = Flask(__name__, static_folder="../static", template_folder="../templates")
    app.config["UPLOAD_FOLDER"] = os.path.abspath("uploads")
    app.config["STAGE_SIZE"] = (960, 640)  # visible stage export size (can change)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    from .web.routes import bp
    from .web.nst_routes import nst_bp  # Add this line
    
    app.register_blueprint(bp)
    app.register_blueprint(nst_bp)  # Add this line
    
    return app
