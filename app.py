from fastapi import FastAPI, UploadFile, File
from model import OnnxModel, ImagePreprocessor
from PIL import Image
import io

app = FastAPI()
pipeline = OnnxModel("resnet18.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_path = io.BytesIO(contents)
    image = ImagePreprocessor().preprocess_numpy(image_path)
    class_id = pipeline.predict(image)
    return {"predicted_class_id": class_id}
