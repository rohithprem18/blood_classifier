from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Initialize FastAPI app
app = FastAPI(title="Blood Group Classifier API")

# Load the trained Keras model
MODEL_PATH = "blood_classifier.keras"
model = load_model(MODEL_PATH)

# Target image size (same as training)
TARGET_SIZE = (128, 128)

# Label mapping (same order as in training)
label_map = {
    'B+': 0, 'O+': 1, 'A+': 2, 'B-': 3,
    'AB-': 4, 'A-': 5, 'O-': 6, 'AB+': 7
}
reverse_label_map = {v: k for k, v in label_map.items()}


# Helper function: preprocess uploaded image
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")


@app.get("/")
def home():
    return {"message": "Welcome to Blood Group Classifier API!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()

        # Preprocess image
        processed_img = preprocess_image(contents)

        # Make prediction
        predictions = model.predict(processed_img)
        class_idx = int(np.argmax(predictions, axis=1)[0])
        blood_group = reverse_label_map[class_idx]
        confidence = float(np.max(predictions) * 100)

        return JSONResponse({
            "filename": file.filename,
            "predicted_blood_group": blood_group,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
