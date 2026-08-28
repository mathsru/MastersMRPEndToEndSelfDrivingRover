#Alex Eliseev
#Script takes the images and labels in data/train and moves them to data/trainbalanced and in the process
#caps the larger classes such as Warning_Mild, Other_Sign which have way too many images as part of them and are creating bias in training. 
#Imports
from pathlib import Path
from collections import defaultdict
import shutil

#Pathing
PROJECT_ROOT = Path(__file__).resolve().parents[3]
#Source folders, moving from train -> train balanced
SRC_LABELS = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "train" / "labels"
SRC_IMAGES = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "train" / "images"
#Destination train balanced
DST_LABELS = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "trainbalanced" / "labels"
DST_IMAGES = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "trainbalanced" / "images"

#Setting restrictions on how many should be moved to trainbalanced
#For now caps only introduced for other signs and warning mild class as they are abundant whilst being very large in diversity and very
#generalized classes
CAPS = {
    81: 2500,   # Other_Sign
    21: 5000    # Warning_Mild (optional example)
}

class_counts = defaultdict(int)
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

#Finds image with matching name to label
def find_image(stem: str):
    for ext in image_extensions:
        p = SRC_IMAGES / f"{stem}{ext}"
        if p.exists():
            return p
    return None

#Goes through for every label and moves it along with the image matching it to train balanced, however for certain classes the amount of them moved
#is tracked and if it hits the cap, they are afterwards no longer moved over and ignored. More specifically they start getting removed from any future
#images.
for label_file in SRC_LABELS.glob("*.txt"):

    lines = label_file.read_text().strip().splitlines()
    if not lines:
        continue

    parsed = []
    image_stem = label_file.stem

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])
        parsed.append((cls, line))

    kept_lines = []
    used_classes = set()

    #Skipping lines with the capped class in them and keeping just the others.
    for cls, line in parsed:

        if cls in CAPS:

            if class_counts[cls] >= CAPS[cls]:
                continue  # skip this box

            class_counts[cls] += 1
            kept_lines.append(line)
            used_classes.add(cls)

        else:
            kept_lines.append(line)
            used_classes.add(cls)

    #If no lines are kept, the image and its label are simply not moved into the balanced training pool and are left behind.
    if not kept_lines:
        continue
    
    #Moving label from train/labels to trainbalanced/labels if cap is not exceeded for the given class
    dst_label = DST_LABELS / label_file.name
    dst_label.write_text("\n".join(kept_lines) + "\n")

    #Moving image from train/images to trainbalanced/images
    img_path = find_image(image_stem)
    if img_path:
        shutil.copy2(img_path, DST_IMAGES / img_path.name)
