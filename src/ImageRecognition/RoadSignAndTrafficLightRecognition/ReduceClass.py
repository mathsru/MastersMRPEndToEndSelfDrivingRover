#Alex Eliseev
#This script is ran to reduce the amount of a certain class in the training pool if needed. In this case it was used to 
#purge the "other sign" class or class 83 which was absolutely overwhelming in quantities and was biasing the entire training pool
#Only every 1/50 instances of the class were kept in both the training and validation label pool.
#Script basically just goes through target folder of labels and any time it encounters class 81 or "other sign" it will delete it 49/50 times
#only exception is if class 81 is the only class in an image then it is spared.
#imports
import os
from pathlib import Path
import random
import shutil

#Keeping only about 1/50 cases of "other class"
#Setting paths
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LABELS_DIR = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "train2" / "labels"

OTHER_SIGN_CLASS = 81
KEEP_RATIO = 1 / 50

for label_file in LABELS_DIR.glob("*.txt"):

    lines = label_file.read_text().strip().splitlines()
    if not lines:
        continue

    new_lines = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])

        # always keep non-81 classes
        if cls != OTHER_SIGN_CLASS:
            new_lines.append(line)
        else:
            # keep only 1/50 of class 81 boxes
            if random.random() < KEEP_RATIO:
                new_lines.append(line)

    # overwrite file (or skip writing if empty)
    if new_lines:
        label_file.write_text("\n".join(new_lines) + "\n")

#Repeating exact same script except for validation labels + images
#Setting paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

"""LABELS_DIR = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "val" / "labels"

OTHER_SIGN_CLASS = 81
KEEP_RATIO = 1 / 50

for label_file in LABELS_DIR.glob("*.txt"):

    lines = label_file.read_text().strip().splitlines()
    if not lines:
        continue

    new_lines = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])

        # always keep non-81 classes
        if cls != OTHER_SIGN_CLASS:
            new_lines.append(line)
        else:
            # keep only 1/50 of class 81 boxes
            if random.random() < KEEP_RATIO:
                new_lines.append(line)

    # overwrite file (or skip writing if empty)
    if new_lines:
        label_file.write_text("\n".join(new_lines) + "\n")"""