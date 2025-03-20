import keras
import numpy as np
from PIL import Image, ImageOps
import os

# Dynamically locate the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'CovidTest.h5')

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

def image_pre(path):
    """ Preprocess the image to match model input requirements. """
    try:
        size = (128, 128)
        image = Image.open(path).convert("L")  # Convert to grayscale
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32) / 255.0  # Normalize
        data = image_array.reshape((-1, 128, 128, 1))  # Reshape for the model
        return data
    except Exception as e:
        print(f"Error processing image: {e}")
        return None  # Return None if an error occurs

def predict(data):
    """ Predict COVID presence from the processed image. """
    if data is None:
        return None
    prediction = model.predict(data)
    return int(np.round(prediction[0][0]))  # Ensure output is 0 or 1
