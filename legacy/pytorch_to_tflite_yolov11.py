"""
cd /path/to/export_folder/
# Activate environment
source /path/to/export_env/bin/activate
python3 pytorch_to_tflite.py # will search for available GPUs automatically and might fail
CUDA_VISIBLE_DEVICES=-1 python3 pytorch_to_tflite.py # force run on CPU to avoid GPU issues
"""


import sys
import os

print("Python executable:", sys.executable)
print("Environment path:", os.environ.get("VIRTUAL_ENV"))
print("Python version:", sys.version)

import numpy
import scipy
import tensorflow as tf
import onnx
import ultralytics

print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("TF:", tf.__version__)
print("ONNX:", onnx.__version__)

ultralytics.checks()

from ultralytics import YOLO

model_path = "/path/to/runs/<run>/weights/best.pt"  # edit: the trained checkpoint to export
print(f"Loading model from: {model_path}")
model_trained = YOLO(model_path)

print("\n--- Model Inspection ---")
print(f"Number of classes: {len(model_trained.names)}")
print(f"Class names: {model_trained.names}")


# Export the model to TFLite format - no quantization
# creates 'yolo11n_float16.tflite', 'yolo11n_float32.tflite' and an ONNX version too
# in the path given above 

results = model_trained.export(
    format="tflite",
    imgsz=320,
    half=True,
    nms=True,
    device="cpu"
)

print(results)

from pathlib import Path

tflite_path = Path(results)
print("TFLite saved at:", tflite_path.resolve())

print("\n--- Exported Model inspection ---")
exported_model = YOLO(tflite_path.resolve())
print(f"Number of classes: {len(exported_model.names)}")
print(f"Class names: {exported_model.names}")
