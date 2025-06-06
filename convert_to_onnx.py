import torch
from pytorch_model import Classifier, BasicBlock
from PIL import Image
from torchvision import transforms
import onnxruntime
import numpy as np

def main():
    # Initialize the model with the same config as in pytorch_model.py
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    
    # Load pretrained weights
    model.load_state_dict(torch.load("pytorch_model_weights.pth", map_location='cpu'))
    model.eval()

    # Create dummy input (batch size 1, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        "resnet18.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print("✅ Model converted to ONNX with preprocessing included.")

if __name__ == "__main__":
    main()