# main.py
import onnxruntime
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json

# Load model
session = onnxruntime.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Preprocessing (same as model.py)
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)

def handler(request):
    """Cerebrium will call this function"""
    body = json.loads(request)

    # Get image base64 from request
    image_data = base64.b64decode(body["image"])
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Preprocess
    preprocessed = preprocess_image(image)

    # Run inference
    outputs = session.run(None, {input_name: preprocessed})
    predicted_class = int(np.argmax(outputs[0]))
    
    return {"class_id": predicted_class}
