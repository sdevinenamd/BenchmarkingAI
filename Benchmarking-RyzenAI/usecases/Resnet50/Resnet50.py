import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
import onnx
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import onnxsim
import shutil
from onnxruntime.quantization import quantize_dynamic, QuantType

class ResnetClassifier:
    def __init__(self, model_name="microsoft/resnet-50"):
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        
    def compile_to_onnx(self, path="models/ONNX/Resnet_Classifier_model.onnx"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            self.model,
            dummy_input,
            path,  # Fixed: was output_path
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        return path  # Fixed: was output_path
        
    def optimize_onnx_model(self, onnx_path, path="models/ONNX/Resnet_Classifier_model_optimized.onnx"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            onnx_model = onnx.load(onnx_path)
            model_simplified, check = onnxsim.simplify(onnx_model)
            
            if check:
                onnx.save(model_simplified, path)  # Fixed: was optimized_path
                print(f"Optimized model saved to {path}")
            else:
                print("Optimization failed")
                
        except ImportError:
            print("onnx-simplifier not available, skipping optimization")
            
        return path  # Fixed: was optimized_path
    
    def quantize_model(self, onnx_path, path="models/ONNX/Resnet_Classifier_model_quantized.onnx"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)   
            quantize_dynamic(
                onnx_path,
                path,  # Fixed: was quantized_path
                weight_type=QuantType.QUInt8
            )
            print(f"Quantized model saved to {path}")
        except Exception as e:
            print(f"Quantization failed: {e}")
            
        return path  # Fixed: was quantized_path

def main():
    classifier = ResnetClassifier()  # Fixed: was CatDogClassifier
    onnx_path = classifier.compile_to_onnx()
    optimized_path = classifier.optimize_onnx_model(onnx_path)
    quantized_path = classifier.quantize_model(optimized_path)
    
    # Verify the models work
    try:
        session = ort.InferenceSession(quantized_path)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = session.run(None, {'input': dummy_input})
        print(f"Model verification successful. Output shape: {output[0].shape}")
    except Exception as e:
        print(f"Model verification failed: {e}")
    
    models = [
        ("Original ONNX", onnx_path),
        ("Optimized", optimized_path), 
        ("Quantized", quantized_path)
    ]
    
    for name, path in models:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"{name:15}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"{name:15}: NOT FOUND")

if __name__ == "__main__":
    main()
