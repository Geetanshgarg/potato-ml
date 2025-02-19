from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
# Add these production configurations
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size 16MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png']

# Load the saved model
model = tf.keras.models.load_model('PatatoTuberDisease_model.h5')

# Define image size that matches your training
IMG_HEIGHT = 180  
IMG_WIDTH = 180   

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image file from request
        file = request.files['image']
        
        # Read and preprocess the image
        nparr = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = np.expand_dims(image_resized, axis=0)
        
        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])
        classes = ['Early Blight', 'Fungi', 'Healthy', 'Late Blight', 'Pest', 'Virus']
        predicted_class_name = classes[predicted_class]
        # Return the prediction
        return jsonify({
            'success': True,
            'predicted_class': predicted_class_name,
            'confidence': float(prediction[0][predicted_class])*100
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)