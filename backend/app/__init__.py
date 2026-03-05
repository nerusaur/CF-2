from flask import Flask

def create_app():
    app = Flask(__name__)

    from app.routes.classify import classify_bp
    app.register_blueprint(classify_bp)

    return app