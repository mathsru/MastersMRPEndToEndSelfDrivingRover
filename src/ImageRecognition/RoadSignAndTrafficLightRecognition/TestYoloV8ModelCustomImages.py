#Alex Eliseev
#This is a script to test a trained YOLOV8 model, takes in a single image and has the model predict classes for it.
#Requires ultralytics to be installed
#imports
from ultralytics import YOLO
import os
from pathlib import Path
import shutil

#Setting paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
#Loading model to be tested
#MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "TrafficLightsOnly_Yolov8__100_epochs_1_720P" / "weights" / "best.pt"
MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "TrafficLightsOnly_Yolov8_60_Epochs_720P" / "weights" / "best.pt"
#Test images path
TEST_IMAGE_PATH = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "test" / "customimages"

model = YOLO(str(MODEL_PATH))

CLASS_NAMES = model.names 
print(model.names)

#Retrieving images from the file
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp",".webp"]
image_files = [
    f for f in TEST_IMAGE_PATH.iterdir()
    if f.suffix.lower() in IMAGE_EXTENSIONS
]

#print(CLASS_NAMES)

#For each image printing identified classes
for image_path in image_files:

    print(f"Processing: {image_path.name}")

    #Run inference
    results = model(str(image_path))

    #Get first result
    result = results[0]

    #Check detections
    if len(result.boxes) == 0:
        print("  No detections found\n")
        continue

    #Loop through detections
    for i, box in enumerate(result.boxes):

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        class_name = CLASS_NAMES[class_id]

        print(
            f"  Detection {i+1}: "
            f"{class_name} "
            f"(confidence={confidence:.3f})"
        )

    print()




