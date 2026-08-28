import os
import sys
import glob
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CARLA_API = PROJECT_ROOT / "src" / "CarlaScripts" / "PythonAPI"

sys.path.append(str(CARLA_API / "carla"))
sys.path.append(str(CARLA_API / "examples"))

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla

# ----------------------------
# YOLO MODEL
# ----------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "YOLOV8TrafficLightAndSignRecognition"
    / "TrafficLightsOnly_Yolov8_60_Epochs_720P"
    / "weights"
    / "best.pt"
)

model = YOLO(str(MODEL_PATH))
model.to("cuda")

CLASS_NAMES = {
    0: "Red_Traffic_Light",
    1: "Yellow_Traffic_Light",
    2: "Green_Traffic_Light",
}

# ----------------------------
# CAMERA
# ----------------------------

IM_WIDTH = 720
IM_HEIGHT = 720

latest_frame = None

# Count valid detections each second
detection_count = 0
last_print_time = time.time()


def process_image(image):
    global latest_frame

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((IM_HEIGHT, IM_WIDTH, 4))
    latest_frame = array[:, :, :3].copy()


# ----------------------------
# CARLA
# ----------------------------

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.load_world("Town05")

spawn_points = world.get_map().get_spawn_points()

print(f"Spawn points: {len(spawn_points)}")

# Visualize spawn point numbers
for i, spawn in enumerate(spawn_points):
    world.debug.draw_string(
        spawn.location + carla.Location(z=1.5),
        str(i),
        color=carla.Color(255, 0, 0),
        life_time=10000.0,
        persistent_lines=True,
    )

START_SPAWN = 199
END_SPAWN = 177

if START_SPAWN >= len(spawn_points):
    raise ValueError("Invalid START_SPAWN")

if END_SPAWN >= len(spawn_points):
    raise ValueError("Invalid END_SPAWN")

start_spawn = spawn_points[START_SPAWN]
end_location = spawn_points[END_SPAWN].location

blueprint = world.get_blueprint_library().filter("model3")[0]

camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(IM_WIDTH))
camera_bp.set_attribute("image_size_y", str(IM_HEIGHT))

vehicle = None
camera = None


def respawn_vehicle():
    global vehicle, camera, latest_frame

    latest_frame = None

    if camera is not None:
        camera.stop()
        camera.destroy()
        camera = None

    if vehicle is not None:
        vehicle.destroy()
        vehicle = None

    while vehicle is None:
        vehicle = world.try_spawn_actor(blueprint, start_spawn)
        if vehicle is None:
            time.sleep(0.1)

    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(
            carla.Location(
                x=2.5,
                z=0.7,
            )
        ),
        attach_to=vehicle,
    )

    camera.listen(process_image)

    # Give sensors time to initialize
    time.sleep(0.5)


respawn_vehicle()

# ----------------------------
# MAIN LOOP
# ----------------------------

try:

    while True:

        # Constant straight driving
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.3,
                steer=0.0,
                brake=0.0,
            )
        )

        # Respawn once vehicle reaches endpoint
        if vehicle.get_location().distance(end_location) < 5.0:
            print("\nReached endpoint - Respawning...\n")
            respawn_vehicle()
            continue

        if latest_frame is not None:

            frame = latest_frame.copy()

            results = model(frame, verbose=False)[0]

            detected = []

            # Keep only detections in middle 55% of image
            margin_x = IM_WIDTH * 0.225
            margin_y = IM_HEIGHT * 0.225

            min_x = margin_x
            max_x = IM_WIDTH - margin_x

            min_y = margin_y
            max_y = IM_HEIGHT - margin_y

            for box in results.boxes:

                cls = int(box.cls)
                conf = float(box.conf)

                if conf < 0.2:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]

                center_x = float((x1 + x2) / 2)
                center_y = float((y1 + y2) / 2)

                if not (
                    min_x <= center_x <= max_x
                    and min_y <= center_y <= max_y
                ):
                    continue

                detected.append((cls, conf))

            detection_count += 1

            current_time = time.time()

            if current_time - last_print_time >= 1.0:
                print(f"Detections this second: {detection_count}")
                detection_count = 0
                last_print_time = current_time

            cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord("q"):
            break

        time.sleep(0.01)

finally:

    if camera is not None:
        camera.stop()
        camera.destroy()

    if vehicle is not None:
        vehicle.destroy()

    cv2.destroyAllWindows()