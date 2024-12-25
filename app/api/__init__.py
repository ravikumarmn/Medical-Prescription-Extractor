from flask import Flask, jsonify
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy"}), 200
    
    from .routes import api_bp
    app.register_blueprint(api_bp)
    
    return app
