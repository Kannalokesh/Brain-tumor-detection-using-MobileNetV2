from flask import Flask, request, render_template
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('best_brain_model.keras')

# Define class labels
class_labels = ['No Tumor found', 'Tumor Detected !!']

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in the request."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    image = load_img(filepath, target_size=(224, 224))  # Resize image to match model input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    # Return prediction result
    return render_template('result.html', prediction=predicted_class, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
