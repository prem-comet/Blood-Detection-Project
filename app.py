from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
import ssl
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.applications.efficientnet  import preprocess_input

# Disable SSL verification if necessary (for downloading model weights etc.)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("models/bg_effnet.keras")

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function
def preprocess_image(file_path):
    """
    Preprocesses the image for model prediction.
    """
    img = load_img(file_path, target_size=(224, 224), color_mode='rgb') # Resize image to model's input size
    img_array = img_to_array(img)                     # Convert image to numpy array
    img_array = preprocess_input(img_array)                    
    img_array = np.expand_dims(img_array, axis=0)    # Add batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')  # Create this template later

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg'}), 400

    filename = secure_filename(file.filename)
    upload_folder = os.path.join('static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)

    file.save(file_path)

    try:
        # Preprocess the image
        img = preprocess_image(file_path)

        # Perform prediction
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))

        class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        predicted_label = class_names[predicted_class]
        confidence = round(float(np.max(predictions[0]) * 100), 2)

        return render_template('index.html',
                               filename=filename,
                               blood_group=predicted_label,
                               confidence=round(float(np.max(predictions[0]) * 100), 2))


    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Remove uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
