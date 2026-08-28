#Alex Eliseev
#This is a script is used for training for training YOLOv8 models for traffic light and sign image recognition. 
#As the project evolves it expands on the amount of datasets added into the training and validation pool.
#imports
from ultralytics import YOLO
import os
from pathlib import Path
import shutil

#Main is needed for multiple workers to be used otherwise crashes
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "TrafficLightsOnly_Yolov8_30_Epochs_720P" / "weights" / "best.pt"
    #MODEL_PATH = PROJECT_ROOT / "src" / "ImageRecognition" / "RoadSignAndTrafficLightRecognition" / "yolov8m.pt"
    model = YOLO(str(MODEL_PATH))
    YAML_PATH = PROJECT_ROOT / "Data" / "TrafficLightsOnly" / "training.yaml"
    
    OUTPUT_DIR = PROJECT_ROOT / "Models" / "YOLOV8TRafficLightAndSignRecognition"

    model.train(
        data=str(YAML_PATH),
        epochs=30,
        imgsz=720,
        batch=16,
        workers=4,
        project=str(OUTPUT_DIR),
        name="TrafficLightsOnly_Yolov8_60_Epochs_720P",
        exist_ok=True
    )
