#Alex Eliseev
#Simple script extracts all traffic light labels and images from the training and validation folders (simply copies)
#and copies them to a matching folder structure in the folder TrafficLights to train a seperate model dedicated to traffic lights because
#after the addition of traffic signs the model simply stopped detecting them at all. With any degree of confidence they simply vanished.
#Script was only ran once
#imports
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

from pathlib import Path
import shutil

#Output and input file paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET_CLASS= {0, 1, 2, 3}
LABELS_DIR = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "trainbalanced" / "labels"
IMAGES_DIR = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "trainbalanced" / "images"

OUTPUT_IMAGES = PROJECT_ROOT / "Data" / "TrafficLightsOnly" / "train" / "images"
OUTPUT_LABELS = PROJECT_ROOT / "Data" / "TrafficLightsOnly" / "train" / "labels"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

copied = 0

for label_file in LABELS_DIR.glob("*.txt"):

    lines = label_file.read_text().strip().splitlines()

    target_class = False

    for line in lines:
        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = int(parts[0])

        if class_id in TARGET_CLASS:
            target_class = True
            break

    if not target_class:
        continue

    # find matching image
    stem = label_file.stem
    img_file = None

    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            img_file = candidate
            break

    if img_file is None:
        print(f"[WARN] Missing image for {label_file.name}")
        continue

    # copy files
    shutil.copy2(img_file, OUTPUT_IMAGES / img_file.name)
    shutil.copy2(label_file, OUTPUT_LABELS / label_file.name)

    copied += 1

print("\n=== DONE ===")
print(f"Copied {copied} samples containing class {TARGET_CLASS}")
print(f"Output folder: {OUTPUT_IMAGES.parent}")

#Repeating process moving all validation images of traffic light classes to new val folder
#Output and input file paths

LABELS_DIR = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "val" / "labels"
IMAGES_DIR = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "val" / "images"

OUTPUT_IMAGES = PROJECT_ROOT / "Data" / "TrafficLightsOnly" / "val" / "images"
OUTPUT_LABELS = PROJECT_ROOT / "Data" / "TrafficLightsOnly" / "val" / "labels"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

copied = 0

for label_file in LABELS_DIR.glob("*.txt"):

    lines = label_file.read_text().strip().splitlines()

    target_class = False

    for line in lines:
        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = int(parts[0])

        if class_id in TARGET_CLASS:
            target_class = True
            break

    if not target_class:
        continue

    # find matching image
    stem = label_file.stem
    img_file = None

    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            img_file = candidate
            break

    if img_file is None:
        print(f"[WARN] Missing image for {label_file.name}")
        continue

    # copy files
    shutil.copy2(img_file, OUTPUT_IMAGES / img_file.name)
    shutil.copy2(label_file, OUTPUT_LABELS / label_file.name)

    copied += 1

print("\n=== DONE ===")
print(f"Copied {copied} samples containing class {TARGET_CLASS}")
print(f"Output folder: {OUTPUT_IMAGES.parent}")


        