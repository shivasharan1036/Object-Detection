from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'temp_uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = 'models/mobilenet_iter_73000.caffemodel'
CONFIG_PATH = 'models/deploy.prototxt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

net = None
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def load_model():
    global net
    
    try:
        print("Loading MobileNet-SSD model...")
        net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if net is not None else "not loaded"
    return jsonify({
        "status": "healthy",
        "service": "ai-backend",
        "model": "MobileNet-SSD",
        "model_status": model_status
    }), 200

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        if net is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        image = cv2.imread(filepath)
        
        if image is None:
            os.remove(filepath)
            return jsonify({"error": "Could not read image"}), 400
        
        height, width, channels = image.shape
        
        detections = perform_detection(image)
        image_with_boxes = draw_boxes(image.copy(), detections)
        
        _, buffer = cv2.imencode('.jpg', image_with_boxes)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f'{base_name}_detected.jpg')
        cv2.imwrite(output_path, image_with_boxes)
        
        json_path = os.path.join(OUTPUT_FOLDER, f'{base_name}_detections.json')
        with open(json_path, 'w') as json_file:
            json.dump(detections, json_file, indent=2)
        
        os.remove(filepath)
        
        response = {
            "detections": detections,
            "image_with_boxes": img_base64,
            "image_dimensions": {
                "width": width,
                "height": height
            },
            "num_objects": len(detections),
            "output_files": {
                "image": output_path,
                "json": json_path
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": "Detection failed", "details": str(e)}), 500

def perform_detection(image, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                  0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections_raw = net.forward()
    
    detections = []
    
    for i in np.arange(0, detections_raw.shape[2]):
        confidence = detections_raw[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            idx = int(detections_raw[0, 0, i, 1])
            box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            detection = {
                "class": CLASSES[idx],
                "confidence": round(float(confidence), 2),
                "bbox": [int(startX), int(startY), int(endX - startX), int(endY - startY)]
            }
            detections.append(detection)
    
    return detections

def draw_boxes(image, detections):
    for detection in detections:
        x, y, w, h = detection['bbox']
        label = f"{detection['class']}: {detection['confidence']:.2f}"
        
        idx = CLASSES.index(detection['class'])
        color = COLORS[idx]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        y_label = y - 15 if y - 15 > 15 else y + 15
        cv2.putText(image, label, (x, y_label), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

if __name__ == '__main__':
    if load_model():
        app.run(host='0.0.0.0', port=5002, debug=True)
    else:
        print("Failed to load model. Please check the model files.")
