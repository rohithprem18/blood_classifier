from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load model
model = load_model("blood_group_cnn_model.keras")

# Same label mapping used in your training
label_map = {'B+': 0, 'O+': 1, 'A+': 2, 'B-': 3, 'AB-': 4, 'A-': 5, 'O-': 6, 'AB+': 7}
reverse_label_map = {i: class_name for class_name, i in label_map.items()}

TARGET_SIZE = (128, 128)

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return None

@app.route('/')
def home():
    return "Blood Classifier Image API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']
    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    if processed_image is None:
        return jsonify({'error': 'Image preprocessing failed'}), 400

    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_blood_group = reverse_label_map[predicted_class_index]
    confidence = float(np.max(predictions) * 100)

    return jsonify({
        'predicted_blood_group': predicted_blood_group,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
