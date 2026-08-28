#Alex Eliseev
#This script is ran only once, for the S2TLD dataset converts XML image annotations -> YOLO txt format.
#Each XML has headers indicating the traffic light type and a bounding box of where it is located in each respective image, script just converts these
#into a class, x_center, y_center, width, height format for YOLOv8 for each traffic light visible/listed in the image/XML file.
#Converts XMLs set aside for testing, training and validation.
#
#imports
import os
import xml.etree.ElementTree as ET
from pathlib import Path

#First converting training XMLs -> training txts
PROJECT_ROOT = Path(__file__).resolve().parents[3]
InputDirectory = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "raw" / "S2TLDDataset" / "OriginalXMLAnnotations" / "train"
OutputDirectory = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "train" / "labels"

#Making output directory if it does not exist
os.makedirs(OutputDirectory,exist_ok=True)

#Class mapping
Classes = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "off": 3
}

#Going through list of XMLs, extracting class of traffic light, its bounding box, image size information and saving in new YOLO format in txt file.
#Saving them to output folder
for xml_file in os.listdir(InputDirectory):
    #Fail safe
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(InputDirectory, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    #Image size
    Width = int(root.find("size/width").text)
    Height = int(root.find("size/height").text)

    YoloLines = []

    #Doing it for every traffic light object (there can be multiple per image)
    for obj in root.findall("object"):

        class_name = obj.find("name").text

        if class_name not in Classes:
            continue

        class_id = Classes[class_name]

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        #Converting to YOLO format (This can always change e.t.c)
        x_center = ((xmin + xmax) / 2) / Width
        y_center = ((ymin + ymax) / 2) / Height
        box_width = (xmax - xmin) / Width
        box_height = (ymax - ymin) / Height

        YoloLines.append(
            f"{class_id} {x_center} {y_center} {box_width} {box_height}"
        )

    #Save txt with same filename
    txt_name = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(OutputDirectory, txt_name)

    #Appending to existing text files if they are already there.
    with open(txt_path, "a") as f:
        f.write("\n".join(YoloLines))

#Repeating same process for the images + annotations set aside for testing
#First converting training XMLs -> training txts
PROJECT_ROOT = Path(__file__).resolve().parents[3]
InputDirectory = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "raw" / "S2TLDDataset" / "OriginalXMLAnnotations" / "test"
OutputDirectory = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "test" / "labels"

#Making output directory if it does not exist
os.makedirs(OutputDirectory,exist_ok=True)

#Class mapping
Classes = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "off": 3
}

#Going through list of XMLs, extracting class of traffic light, its bounding box, image size information and saving in new YOLO format in txt file.
#Saving them to output folder
for xml_file in os.listdir(InputDirectory):
    #Fail safe
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(InputDirectory, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    #Image size
    Width = int(root.find("size/width").text)
    Height = int(root.find("size/height").text)

    YoloLines = []

    #Doing it for every traffic light object (there can be multiple per image)
    for obj in root.findall("object"):

        class_name = obj.find("name").text

        if class_name not in Classes:
            continue

        class_id = Classes[class_name]

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        #Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / Width
        y_center = ((ymin + ymax) / 2) / Height
        box_width = (xmax - xmin) / Width
        box_height = (ymax - ymin) / Height

        YoloLines.append(
            f"{class_id} {x_center} {y_center} {box_width} {box_height}"
        )

    # save txt with same filename
    txt_name = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(OutputDirectory, txt_name)

    with open(txt_path, "a") as f:
        f.write("\n".join(YoloLines))

#Lastly repeating process for images + annotations set aside for the validation during training
#First converting training XMLs -> training txts
PROJECT_ROOT = Path(__file__).resolve().parents[3]
InputDirectory = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "raw" / "S2TLDDataset" / "OriginalXMLAnnotations" / "val"
OutputDirectory = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "val" / "labels"

#Making output directory if it does not exist
os.makedirs(OutputDirectory,exist_ok=True)

#Class mapping
Classes = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "off": 3
}

#Going through list of XMLs, extracting class of traffic light, its bounding box, image size information and saving in new YOLO format in txt file.
#Saving them to output folder
for xml_file in os.listdir(InputDirectory):
    #Fail safe
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(InputDirectory, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    #Image size
    Width = int(root.find("size/width").text)
    Height = int(root.find("size/height").text)

    YoloLines = []

    #Doing it for every traffic light object (there can be multiple per image)
    for obj in root.findall("object"):

        class_name = obj.find("name").text

        if class_name not in Classes:
            continue

        class_id = Classes[class_name]

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        #convert to YOLO format
        x_center = ((xmin + xmax) / 2) / Width
        y_center = ((ymin + ymax) / 2) / Height
        box_width = (xmax - xmin) / Width
        box_height = (ymax - ymin) / Height

        YoloLines.append(
            f"{class_id} {x_center} {y_center} {box_width} {box_height}"
        )

    # save txt with same filename
    txt_name = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(OutputDirectory, txt_name)

    with open(txt_path, "a") as f:
        f.write("\n".join(YoloLines))