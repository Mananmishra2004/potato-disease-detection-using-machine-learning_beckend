from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os
import requests
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ CORS FIX
CORS(app, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# 🔥 GOOGLE DRIVE MODEL DOWNLOAD
MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "potato_disease_model1.h5")

try:
    # Create model folder if not exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded!")

    # Load model
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# --- Constants ---
IMG_WIDTH, IMG_HEIGHT = 256, 256

CLASS_NAMES = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy'
]


def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/')
def home():
    return "<h1>Potato Disease Detection API is running 🚀</h1>"


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():

    if request.method == 'OPTIONS':
        return '', 200

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        img_processed = prepare_image(img_bytes)

        preds = model.predict(img_processed)

        predicted_class_index = int(np.argmax(preds))
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(preds))

        return jsonify({
            'disease': predicted_class_name,
            'confidence': confidence,
            'probabilities': preds[0].tolist()
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# Local testing only
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)