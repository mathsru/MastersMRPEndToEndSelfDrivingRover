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
    MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "Traffic&Signs_Yolov8__100_epochs_1_720P" / "weights" / "best.pt"
    model = YOLO(str(MODEL_PATH))
    YAML_PATH = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "balancedtraining.yaml"
    
    OUTPUT_DIR = PROJECT_ROOT / "Models" / "YOLOV8TRafficLightAndSignRecognition"

    model.train(
        data=str(YAML_PATH),
        epochs=20,
        imgsz=720, #Much increased resolution very valuable for text such as speed limits and minor differences between U-turn and left turn
        batch=16,
        workers=4,
        project=str(OUTPUT_DIR),
        name="Traffic&Signs_Yolov8__120_epochs_1_720P",
        exist_ok=True
    )
