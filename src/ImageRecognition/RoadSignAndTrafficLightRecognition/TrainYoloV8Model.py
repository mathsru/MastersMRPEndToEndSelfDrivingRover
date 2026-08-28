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
    MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "traffic_lights_and_signs_attempt_1_40_epochs" / "weights" / "best.pt"
    model = YOLO(str(MODEL_PATH))
    YAML_PATH = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "trafficlightdata.yaml"
    
    OUTPUT_DIR = PROJECT_ROOT / "Models" / "YOLOV8TRafficLightAndSignRecognition"

    model.train(
        data=str(YAML_PATH),
        epochs=25,
        imgsz=720,
        batch=16,
        workers=4,
        project=str(OUTPUT_DIR),
        name="traffic_lights_and_signs_attempt_1_65_epochs",
        exist_ok=True
    )
