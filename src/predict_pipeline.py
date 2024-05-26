import numpy as np
from tensorflow.keras.models import load_model 
from PIL import Image

def load_trained_model(model_path):
    """Load the trained model."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading the trained model: {e}")
        return None

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess the image for prediction."""
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict(image_path, model):
    """Make predictions on the image."""
    try:
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is not None:
            prediction = model.predict(preprocessed_image)
            return prediction
        else:
            return None
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None