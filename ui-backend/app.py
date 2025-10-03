from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import os
from werkzeug.utils import secure_filename
import io
import base64
from PIL import Image

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
AI_BACKEND_URL = os.getenv('AI_BACKEND_URL', 'http://ai-backend:5002')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "ui-backend"}), 200

@app.route('/detect', methods=['POST'])
def detect_objects():
    """
    Endpoint to upload image and get object detection results
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        with open(filepath, 'rb') as f:
            files = {'image': (filename, f, 'image/jpeg')}
            response = requests.post(f'{AI_BACKEND_URL}/detect', files=files, timeout=60)
        
        os.remove(filepath)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify(result), 200
        else:
            return jsonify({"error": "AI backend error", "details": response.text}), 500
            
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to connect to AI backend", "details": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/detect-and-save', methods=['POST'])
def detect_and_save():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        with open(filepath, 'rb') as f:
            files = {'image': (filename, f, 'image/jpeg')}
            response = requests.post(f'{AI_BACKEND_URL}/detect', files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            import json
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(UPLOAD_FOLDER, f'{base_name}_detections.json')
            
            with open(json_path, 'w') as json_file:
                json.dump(result['detections'], json_file, indent=2)
            
            if 'image_with_boxes' in result:
                img_data = base64.b64decode(result['image_with_boxes'])
                output_img_path = os.path.join(UPLOAD_FOLDER, f'{base_name}_detected.jpg')
                with open(output_img_path, 'wb') as img_file:
                    img_file.write(img_data)
                
                result['saved_files'] = {
                    'json': json_path,
                    'image': output_img_path
                }
            
            return jsonify(result), 200
        else:
            os.remove(filepath)
            return jsonify({"error": "AI backend error", "details": response.text}), 500
            
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)