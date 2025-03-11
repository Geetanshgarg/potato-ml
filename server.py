from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import time
import logging

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure app for production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the ML model
MODEL = None
CLASSES = ['Bacteria', 'Early Blight', 'Fungi', 'Healthy', 'Late Blight', 'Pest', 'Virus']

def load_model():
    global MODEL
    model_path = os.environ.get('MODEL_PATH', 'models/PatatoTuberDisease_model.keras')
    
    try:
        logger.info(f"Loading mode l from {model_path}")
        start_time = time.time()
        MODEL = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        MODEL = None
        raise

# Preprocess image for the model
def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in app.config['UPLOAD_EXTENSIONS']:
        return jsonify({'error': f'Unsupported file extension. Use {app.config["UPLOAD_EXTENSIONS"]}'}), 400
    
    try:
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = MODEL.predict(processed_image)
        prediction_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][prediction_idx])
        predicted_class = CLASSES[prediction_idx]
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'class_id': int(prediction_idx)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': CLASSES})

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))