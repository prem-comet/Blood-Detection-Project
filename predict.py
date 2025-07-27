import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

model = tf.keras.models.load_model('./models/model.h5')
file_path='static/uploads/cluster_3_1307.jpg'

def preprocess_image(file_path):
    """
    Preprocesses the image for model prediction.
    """
    img = load_img(file_path, target_size=(100, 100), color_mode='grayscale') # Resize image to model's input size
    img_array = img_to_array(img)                     # Convert image to numpy array
    img_array = img_array / 255.0                    
    img_array = np.expand_dims(img_array, axis=0)    # Add batch dimension
    return img_array

def predict():
        # Preprocess the image
        img = preprocess_image(file_path)

        # Perform prediction
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        print('predicted_class is :', predicted_class)

        # Define class labels
        class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        predicted_label = class_names[predicted_class]

        # Return the result
        res={
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': float(np.max(predictions[0]))
        }
        print(res)
if __name__ == '__main__':
    predict()