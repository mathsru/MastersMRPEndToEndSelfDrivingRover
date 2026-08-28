#Alex Eliseev
#Simple script goes through the databalanced/train folder and simply counts the quantity of each class in the training pool.
#Classes that are underrepresented in the training pool can be expected to be identified poorly by the model or even mislabeled as other classes.
#These classes should either be removed entirely later in the project or have new images be added to pad their counts.
#imports
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

#Pathing
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS_DIR = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "trainbalanced" / "labels"

#Class dictionairy
CLASS_NAMES = {
    0: "Red_Traffic_Light",
    1: "Yellow_Traffic_Light",
    2: "Green_Traffic_Light",
    3: "Turned_Off_Traffic_Light",
    4: "Stop_Sign",
    5: "Stop_Sign_Ahead",
    6: "Traffic_Light_Ahead",
    7: "Cars_Banned_From_Lane",
    8: "Non_Cars_Banned_From_Lane",
    9: "Dead_End",
    10: "No_Entry",
    11: "Road_Closed",
    12: "Yield",
    13: "Yield_Sign_Ahead",
    14: "Yield_Pedestrians",
    15: "Yield_Bicycles",
    16: "Railroad_Crossing",
    17: "Non_Car_Path_Dont_Go_Here",
    18: "Roundabout_Ahead",
    19: "Detour",
    20: "Lane_Merges_Ahead",
    21: "Warning_Mild",
    22: "Warning_Considerate",
    23: "Warning_Severe",
    24: "Descent",
    25: "Intersection_Ahead",
    26: "Winding_Road",
    27: "Construction",
    28: "No_Passing_Zone",
    29: "Island_Ahead_Pass_On_Left_Or_Right",
    30: "Passing_Allowed",
    31: "Parking",
    32: "No_Parking",
    33: "Reserved_Parking",
    34: "No_Stopping",
    35: "No_Honking",
    36: "Turn_Any_Direction",
    37: "Only_Go_Straight",
    38: "LeftLaneLeftMiddleLaneStraightRightLaneRight",
    39: "Ramp_Chevron_Left",
    40: "Ramp_Chevron_Right",
    41: "Turn_Keep_Left",
    42: "Turn_Keep_Right",
    43: "Turn_Left_Or_Right",
    44: "LeftLaneLeftRightLaneRight",
    45: "Turn_Left_Or_Go_Straight",
    46: "Turn_Right_Or_Go_Straight",
    47: "Stay_Left_To_Go_Straight_Or_Both_Lanes_Right",
    48: "Stay_Right_To_Go_Straight_Or_Both_Lanes_Left",
    49: "No_Left_Turn",
    50: "No_Right_Turn",
    51: "No_Right_Turn_On_Red",
    52: "No_Left_Turn_Or_Straight",
    53: "No_Right_Turn_Or_Straight",
    54: "No_Left_Turn_Or_Right_Turn",
    55: "No_U_Turn",
    56: "U_Turn_Allowed",
    57: "Dual_Speed_Limit",
    58: "Speed_Limit_100",
    59: "Speed_Limit_110",
    60: "Speed_Limit_120",
    61: "Speed_Limit_130",
    62: "Speed_Limit_20",
    63: "Speed_Limit_30",
    64: "Speed_Limit_40",
    65: "Speed_Limit_50",
    66: "Speed_Limit_60",
    67: "Speed_Limit_70",
    68: "Speed_Limit_80",
    69: "Speed_Limit_90",
    70: "End_Of_Speed_Limit_Not_For_Canada",
    71: "Set_School_Children_Zone",
    72: "End_Of_School_Children_Zone",
    73: "Clearance",
    74: "No_High_Beams",
    75: "Airport",
    76: "Hospital",
    77: "Exit",
    78: "Road_Bump",
    79: "Road_Dip",
    80: "Unpaved_Road",
    81: "Other_Sign"
}


class_counts = defaultdict(int)

for label_file in LABELS_DIR.glob("*.txt"):

    lines = label_file.read_text().strip().splitlines()

    for line in lines:
        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        class_counts[class_id] += 1

print("\n=== CLASS DISTRIBUTION (BOX COUNTS) ===\n")

for class_id in sorted(class_counts):
    name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
    print(f"{class_id:2d} | {name:50s} | {class_counts[class_id]}")

import matplotlib.pyplot as plt

# Prepare data for plotting
class_ids = sorted(class_counts.keys())
class_names = [CLASS_NAMES.get(class_id, f"Unknown_{class_id}") for class_id in class_ids]
counts = [class_counts[class_id] for class_id in class_ids]

# Create bar chart
plt.figure(figsize=(20, 8))

plt.bar(class_names, counts)

plt.xlabel("Class")
plt.ylabel("Number of Labels")
plt.title("YOLO Training Dataset Class Distribution")

# Rotate labels so they are readable
plt.xticks(rotation=90, fontsize=8)

plt.tight_layout()

plt.show()
        