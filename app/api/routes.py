import io
import base64
from PIL import Image
from flask import Blueprint, request, jsonify
from .utils import process_prescription, ProcessingError

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@api_bp.route('/process', methods=['POST'])
def process_image():
    """
    Process prescription image and return OCR and structured data.
    
    Accepts two formats:
    1. Multipart form data with 'image' field containing the prescription image
    2. JSON with base64-encoded image in 'image' field
    
    Returns:
    - JSON object containing OCR text and structured data
    """
    try:
        image = None
        
        # Handle multipart form data
        if 'image' in request.files:
            image_file = request.files['image']
            if not image_file.filename:
                return jsonify({"error": "Empty image file provided"}), 400
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
        # Handle base64 image data
        elif request.is_json and 'image' in request.json:
            base64_image = request.json['image']
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))
            
        else:
            return jsonify({"error": "No image provided. Send either a file upload or base64 image data"}), 400
            
        # Process the prescription
        result = process_prescription(image)
        
        return jsonify(result), 200
        
    except ProcessingError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
