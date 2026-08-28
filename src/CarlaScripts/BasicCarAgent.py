#Alex Eliseev
#Basic Carla script that spawns a car that automatically drives around the map and collects car footage using a camera. The recording is saved to
#custom videos so that the image recognition model can be tested on it.
#Goal is to have the script simply record and then save some footage, then test the image recognition model on the footage
#Upgrade is to integrate the script directly into the carla script itself, all be it this is much harder
#Imports
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_API = PROJECT_ROOT / "src" / "CarlaScripts" / "PythonAPI" 
sys.path.append(os.path.join(CARLA_API, "carla"))
sys.path.append(os.path.join(CARLA_API, "examples"))

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from agents.navigation.basic_agent import BasicAgent

#Function processes images recorded by the camera in the world/simulation
def process_img(image):
    global video_writer

    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    frame = i2[:, :, :3]

    cv2.imshow("Camera", frame)
    cv2.waitKey(1)

    if video_writer is not None:
        video_writer.write(frame)

    return frame

actor_list = []
video_writer = None
#Using image width/height of my model training
IM_WIDTH = 720
IM_HEIGHT = 720
OUTPUT_FOLDER_VIDEO_RECORDING = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "test" / "customvideos" / "VideoTown05.mp4"

#Tries to run the code under the try and in case of an error or by reaching the end destroys all actors and cuts the connection of the script to the server
try:
    client = carla.Client('localhost', 2000) #Connecting to locally hosted carla server/world
    client.set_timeout(4.0)

    #Loading world X unto the server
    world = client.load_world('Town05')
    world = client.get_world()
    time.sleep(5) #Delay to give the server a moment to load the world

    #Loading a Tesla Model 3
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0] #Selecting a car to load

    #Spawning car
    spawn_point = random.choice(world.get_map().get_spawn_points()) #Randomly selecting spawn point
    vehicle = world.spawn_actor(bp, spawn_point)
    actor_list.append(vehicle)

    video_writer = cv2.VideoWriter(
        OUTPUT_FOLDER_VIDEO_RECORDING,
        cv2.VideoWriter_fourcc(*'mp4v'),
        20.0,
        (IM_WIDTH, IM_HEIGHT)
    )

    #Camera setup
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    camera_bp.set_attribute('fov', '110')
    camera_transform = carla.Transform(
        carla.Location(x=2.5, z=0.7)
    )
    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )
    actor_list.append(camera)

    #Start recording
    camera.listen(process_img)

    #Turning the car into our actor/agent that will automaticlaly drive around the map and stop at things like red lights, stop signs e.t.c
    #Automatic driving is built into the Carla source package of scripts, camera is attached to the car and records right away.
    agent = BasicAgent(vehicle)
    agent.set_target_speed(30)
    time.sleep(5)
    agent.ignore_traffic_lights(True)
    agent.ignore_stop_signs(True)
    spawn_points = world.get_map().get_spawn_points()
    destination = random.choice(spawn_points).location
    agent.set_destination(destination)
    DestinationReached = 0

    accelerating = True
    switch_time = time.time() + 10

    while DestinationReached < 2:

        if accelerating:
            agent.set_target_speed(40)

            if time.time() > switch_time:
                accelerating = False

        else:
            agent.set_target_speed(0)

        control = agent.run_step()
        vehicle.apply_control(control)

        if agent.done():

            DestinationReached += 1

            print(f"Reached destination {DestinationReached}/2")

            if DestinationReached < 2:
                destination = random.choice(
                    world.get_map().get_spawn_points()
                ).location

                agent.set_destination(destination)

                accelerating = True
                switch_time = time.time() + 10

        time.sleep(0.05)

finally:
    if camera is not None:
        camera.stop()
    #Releasing video writer
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
