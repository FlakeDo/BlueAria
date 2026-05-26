import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the path of the parent folder (BlueAria/) to the path, to be able to import data and signal

from ultralytics import YOLO

from yolo_data import get_YOLO_data


yaml_file = get_YOLO_data("yaml_WhaleSounds_HuggingFace_dataset")

model = YOLO("yolo26n.yaml")
print("Data loaded, model ready.")

results = model.train(data = yaml_file, epochs = 5)
print("Training complete.")

model.export(format = "onnx")
print("Model exported.")

