from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your .keras model
model = load_model("blood_group_cnn_model.keras")

@app.route('/')
def home():
    return "Blood Classifier API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        input_data = np.array(data)
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
