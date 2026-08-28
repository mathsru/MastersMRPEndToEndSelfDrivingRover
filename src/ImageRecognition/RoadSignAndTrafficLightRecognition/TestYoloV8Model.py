#Alex Eliseev
#This is a script to test a trained YOLOV8 model using the test dataset
#Requires ultralytics to be installed
#imports
from ultralytics import YOLO
import os
from pathlib import Path
import shutil

#Setting paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
#Loading model to be tested
MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "Traffic&Signs_Yolov8__40_epochs_1_720P" / "weights" / "best.pt"
#Test images path
TEST_IMAGES_PATH = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "test" / "images"
TEST_LABELS_PATH = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "test" / "labels"

model = YOLO(str(MODEL_PATH))

CLASS_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "off"
}

def read_labels(label_path):
    """Returns list of ground truth class IDs"""
    if not label_path.exists():
        return []
    
    classes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                classes.append(int(parts[0]))
    return classes


def predict_classes(image_path):
    results = model.predict(source=str(image_path), conf=0.4, verbose=False)

    predicted_classes = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            predicted_classes.append(cls_id)

    return predicted_classes


correct = 0
total = 0

for img_path in sorted(TEST_IMAGES_PATH.glob("*.jpg")):

    label_path = TEST_LABELS_PATH / (img_path.stem + ".txt")

    gt_classes = read_labels(label_path)
    pred_classes = predict_classes(img_path)

    # SIMPLE CHECK: does red exist correctly?
    gt_has_red = 0 in gt_classes
    pred_has_red = 0 in pred_classes
    print(pred_has_red)

    is_correct = (gt_has_red == pred_has_red)

    print(f"{img_path.name}: {'✔' if is_correct else '❌'}")

    correct += int(is_correct)
    total += 1

print("\n--- SUMMARY ---")
print(f"Accuracy: {correct/total:.3f}")




