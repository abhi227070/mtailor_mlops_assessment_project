import onnxruntime as ort
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def preprocess_numpy(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # batch dim
        return img.astype(np.float32) 

class OnnxModel:
    def __init__(self, onnx_path='resnet18.onnx'):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, preprocessed_image):
        outputs = self.session.run(None, {self.input_name: preprocessed_image})
        preds = outputs[0]
        class_id = int(np.argmax(preds, axis=1)[0])
        return class_id

class ModelPipeline:
    def __init__(self, onnx_path='resnet18.onnx'):
        self.preprocessor = ImagePreprocessor()
        self.model = OnnxModel(onnx_path)

    def predict_from_path(self, image_path):
        img = self.preprocessor.preprocess_numpy(image_path)
        return self.model.predict(img)
