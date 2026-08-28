#Alex Eliseev
#This is a script to test a trained YOLOV8 model, loads videos and parses them frame by frame and has the model identify red/yellow/green traffic lights.
#Requires ultralytics to be installed
#imports
from ultralytics import YOLO
from pathlib import Path
import cv2
import time

#Setting paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
#Loading model to be tested
MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "Traffic&Signs_Yolov8__100_epochs_1_720P" / "weights" / "best.pt"
MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "TrafficLightsOnly_Yolov8_30_Epochs_720P" / "weights" / "best.pt"
#Test images path
TEST_VIDEO_PATH = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "test" / "customvideos"

model = YOLO(str(MODEL_PATH))


CLASS_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "off"
}

VIDEO_FOLDER = TEST_VIDEO_PATH
# Supported video types
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# Get all videos
VideoFiles = [
    file for file in VIDEO_FOLDER.iterdir()
    if file.suffix.lower() in VIDEO_EXTENSIONS
]

# Loop through all videos
for VideoPath in VideoFiles:
    start = time.time()
    #Open video
    Capture = cv2.VideoCapture(str(VideoPath))
    FrameCounter = 0

    while Capture.isOpened():

        ret, frame = Capture.read()

        # End of video
        if not ret:
            break

        # Process every 3rd frame
        if FrameCounter % 5 == 0:
            # DISPLAY FRAME


            # Run YOLO inference
            results = model(frame)

            result = results[0]

            # Check detections
            if len(result.boxes) > 0:

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = CLASS_NAMES.get(class_id, "unknown")

                    print(
                        f"Detected class {class_name} "
                        f"with confidence {confidence:.3f}"
                    )
                    cv2.imshow("Current Frame", frame)

                    print("Detection found. Press SPACE to continue, Q to quit.")

                    while True:
                        key = cv2.waitKey(0) & 0xFF

                        if key == ord(' '):      # Spacebar
                            break                # Continue processing video

                        elif key == ord('q'):
                            Capture.release()
                            cv2.destroyAllWindows()
                            exit()

            else:
                print("No detections")

        FrameCounter += 1

    Capture.release()
    elapsed = time.time() - start
    print("Processing time: ",elapsed)
    print("Finished processing video")






