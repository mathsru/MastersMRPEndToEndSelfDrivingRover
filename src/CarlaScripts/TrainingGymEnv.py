#Alex Eliseev
#This script sets up the training environment to get the agent to brake for red lights and stop signs around the city. The agent fully controls the speed but, does not have control of steering yet.
#Imports
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque #This is needed to make the prioritized replay.
import random
import copy
from ultralytics import YOLO
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

#Loading Yolo traffic light recognition model
MODEL_PATH = PROJECT_ROOT / "Models" / "YOLOV8TrafficLightAndSignRecognition" / "TrafficLightsOnly_Yolov8_60_Epochs_720P" / "weights" / "best.pt"
TrafficLightModel = YOLO(str(MODEL_PATH))
TrafficLightModel.to("cuda")

#Function for accessing the current frame view of the camera
LatestFrame = None
IM_WIDTH = 720
IM_HEIGHT = 720
def process_image(image):
    global LatestFrame
    array = np.frombuffer(image.raw_data,dtype=np.uint8)
    array = array.reshape(IM_HEIGHT,IM_WIDTH,4)
    LatestFrame = array[:,:,:3].copy()
    #cv2.imshow("Camera",LatestFrame)
    #cv2.waitKey(1)

#Setting up GYM environment for Stable Baselines 3 library of RL
import gymnasium as gym
from gymnasium import spaces
class CarlaGymEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(10.0)
        self.client.load_world("Town05")
        self.world = self.client.get_world()
        if "Town05" not in self.world.get_map().name:
            self.world = self.client.load_world("Town05")
        self.actor_list = []
        self.warmup_time = 5.0
        self.episode_start_time = None
        #Car will have 2 choices of actions that it can make, reduce speed to 0 or increase speed to 30
        # 0 = do nothing/dont brake
        # 1 = brake
        self.action_space = spaces.Discrete(2)
        self.collision = False
        self.episode_reward = 0
        self.episode_number = 1
        self.log_file = (
            PROJECT_ROOT
            / "Models"
            / "ReinforcementLearningModelsCarla"
            / "PPO_Braking"
            / "training_log.csv"
        )
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("Episode,Reward,Reason\n")
        #Observation space (what is being fed into the model as input features)
        self.observation_space = spaces.Dict({

            "image": spaces.Box(
                low=0,
                high=255,
                shape=(128, 128, 3),
                dtype=np.uint8
            ),

            "speed": spaces.Box(
                low=0,
                high=120,
                shape=(1,),
                dtype=np.float32
            ),

            "red_light": spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "current_target_speed": spaces.Box(
                low=0,
                high=100,
                shape=(1,),
                dtype=np.float32
            )
        })
        self.vehicle = None
        self.agent = None
        self.stuck_counter = 0
        self.target_speed = 30

    #Function to start a new training episode
    def reset(self,seed=None,options=None):
        self.collision = False
        self.episode_reward = 0
        global LatestFrame
        LatestFrame = None
        super().reset(seed=seed)
        self.cleanup()
        self.stuck_counter = 0
        #Getting car for training
        Blueprint = self.world.get_blueprint_library().filter("model3")[0]
        #Setting inital spawn point
        spawn = random.choice(self.world.get_map().get_spawn_points())
        #Spawning
        self.vehicle = self.world.spawn_actor(Blueprint,spawn)
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0
            )
        )

        time.sleep(2)
        self.episode_start_time = time.time()
        self.agent = BasicAgent(self.vehicle)
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=0.0,
                steer=0.0
            )
        )
        self.actor_list.append(self.vehicle)
        #Attaching camera to car
        camera_bp = (
            self.world
            .get_blueprint_library()
            .find("sensor.camera.rgb")
        )
        camera_bp.set_attribute(
            "image_size_x",
            str(IM_WIDTH)
        )
        camera_bp.set_attribute(
            "image_size_y",
            str(IM_HEIGHT)
        )
        camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(
                    x=2.5,
                    z=0.7
                )
            ),
            attach_to=self.vehicle
        )
        camera.listen(process_image)
        #Collision sensor
        collision_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )

        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )

        collision_sensor.listen(self.process_collision)

        self.actor_list.append(collision_sensor)
        self.actor_list.append(camera)
        time.sleep(1)
        self.agent.ignore_traffic_lights(True)
        self.agent.ignore_stop_signs(True)
        self.warmup_end_time = time.time() + 5
        #Choosing destination spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        destination = random.choice(spawn_points).location
        while destination.distance(self.vehicle.get_location()) < 50:
            destination = random.choice(spawn_points).location
        self.agent.set_destination(destination)
        self.target_speed = 30

        while LatestFrame is None:
            time.sleep(0.01)


        observation = self.get_state()
        return observation,{}

    #Function executes an action and sees the reward for it
    def step(self, action):
        terminated = False
        # Check if previous route finished FIRST
        if self.agent.done():

            #print("Destination reached!")

            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0,
                    hand_brake=True
                )
            )

            time.sleep(0.2)

            observation = self.get_state()
            reward = 250
            self.episode_reward += reward
            with open(self.log_file, "a") as f:
                f.write(
                    f"{self.episode_number},"
                    f"{self.episode_reward:.2f},"
                    f"Destination\n"
                )
            self.episode_number += 1
            return observation, reward, True, False, {}

        if time.time() < self.warmup_end_time:
            control = self.agent.run_step()

        else:
            #print(action)
            if action == 0:
                self.target_speed = 0
            elif action == 1:
                self.target_speed = 30

            self.agent.set_target_speed(self.target_speed)

            control = self.agent.run_step()


        self.vehicle.apply_control(control)
        observation = self.get_state()
        #Respawning the vehicle if it crashes, collisions happen often this way, their training damage is mitigated
        if self.collision:
            self.respawn_vehicle()
            self.collision = False

        reward = self.calculate_reward(observation)
        self.episode_reward += reward
        velocity = self.vehicle.get_velocity().length() * 3.6
        red_light = observation["red_light"][0]
        if velocity < 1 and red_light == 0:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 400: #Similarily if vehicle stuck reset the car but, dont scratch the episode
            self.respawn_vehicle()
            self.stuck_counter = 0
            self.collision = False
        truncated = False
        return observation, reward, terminated, truncated, {}

    #Function returns the current state of the vehicle, i.e the speed its currently travelling it and if it has reached the destination
    #Also if the camera sees red lights, stop signs e.t.c
    def get_state(self):
        #Current speed
        velocity = (self.vehicle.get_velocity().length() * 3.6) #Speed in km/h

        #Current camera view
        global LatestFrame
        frame = LatestFrame

        #See if its a red light with Yolo model
        red_light = 0

        if frame is None:
            frame = np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)

        
        results = TrafficLightModel(frame, verbose=False)[0]
        r1_frame = cv2.resize(frame,(128,128))

        #Camera center
        frame_center_x = IM_WIDTH / 2
        frame_center_y = IM_HEIGHT / 2
        #Camera should only detect reds in the middle of the screen and not detect them by accident on the sides
        # 0.25 means middle 50% of image
        center_threshold_x = IM_WIDTH * 0.25
        center_threshold_y = IM_HEIGHT * 0.25
        steering = self.vehicle.get_control().steer
        driving_straight = abs(steering) < 0.1

        for box in results.boxes:

            cls = int(box.cls)
            confidence = float(box.conf)

            if cls != 0:
                continue
            if confidence < 0.2:
                continue
            #Red light detection location
            x1, y1, x2, y2 = box.xyxy[0]

            object_center_x = float((x1 + x2) / 2)
            object_center_y = float((y1 + y2) / 2)
            #Making sure detected red light in the middle of the view
            in_center = (
                abs(object_center_x - frame_center_x) < center_threshold_x
                and
                abs(object_center_y - frame_center_y) < center_threshold_y
            )
            if in_center and driving_straight:
                red_light = 1
                break

        return {
            "image": r1_frame,
            "speed": np.array([velocity], dtype=np.float32),
            "red_light": np.array([red_light], dtype=np.float32),
            "current_target_speed": np.array([self.target_speed],dtype=np.float32)
        }

    #Function calculates reward based off of the current state
    def calculate_reward(self,observation):
        speed = observation["speed"][0]
        red_light = observation["red_light"][0]
        reward = 0

        #if red_light == 1:
            #print("YOLO: RED LIGHT DETECTED")

        #Car gets rewarded for moving towards its destination, for now faster = better
        if red_light == 1 and self.is_driving_straight(): #Rewearding for having lower speed/stopping when red light is detected, not punishing if red light seen during a turn
            if speed < 0.1:
                reward += 2500 #Massive reward for stopping on a red
            elif speed < 1: #Small reward for substantially slowing down
                reward +=1500
            elif speed < 2: #Small reward for substantially slowing down
                reward +=500
            elif speed < 3: #Lesser epnalty for running red on a lower speed
                reward +=100
            elif speed < 4: #Small reward for substantially slowing down
                reward -= 50
            elif speed < 5: #Small reward for substantially slowing down
                reward -=100
            elif speed < 10: #Small reward for substantially slowing down
                reward -= 250
            else:
                reward -= 500
        else: 
            if speed < 0.5: #Car should be moving forwards
                reward -= 1 #Large penalty for standing on a green, prevents model from waitinf for red to come back
            elif speed < 5:
                reward += 0.1 #Small passive reward for moving
            elif speed <= 10:
                reward += 0.25 #Slightly larger reward for driving
            elif speed <= 20:
                reward += 0.5
            else:
                reward += 3 #Larger reward for driving fast, but all lost when red is ran

        #print(f"Speed: {speed:.2f} km/h | Reward: {reward}")
        #print(self.is_driving_straight())

        return reward

    #Cleanup function
    def cleanup(self):

        #print("Cleanup called.")

        for actor in self.actor_list:

            try:
                if actor is not None:

                    if "sensor" in actor.type_id:
                        actor.stop()

                    destroyed = actor.destroy()

                    #print(
                    #    "Destroyed:",
                    #    actor.type_id,
                    #   destroyed
                    #)

            except Exception as e:
                print("Cleanup error:", e)


        self.actor_list.clear()

        self.vehicle = None
        self.agent = None

        time.sleep(1)

    #Function returns if the car is driving straight, the model should not be punished when seeing a red of the opposing traffic during a turn
    def is_driving_straight(self):
        control = self.vehicle.get_control()

        #CARLA steering range:
        # -1 = full left
        #  0 = straight
        # +1 = full right

        if abs(control.steer) < 0.1:
            return True

        return False

    #Function detects if the car has collided
    def process_collision(self, event):
        #print("COLLISION DETECTED")
        self.collision = True

    #Respawns the vehicle a mini reset without ending the episode
    def respawn_vehicle(self):

        print("Respawning vehicle...")
        # Remove old vehicle, camera, collision sensor
        self.cleanup()
        self.collision = False
        # Get vehicle blueprint
        blueprint = self.world.get_blueprint_library().filter("model3")[0]
        # Find a valid spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        while True:
            spawn = random.choice(spawn_points)
            self.vehicle = self.world.try_spawn_actor(
                blueprint,
                spawn
            )
            if self.vehicle is not None:
                break
        print("Vehicle respawned")
        self.actor_list.append(self.vehicle)
        # Reset vehicle controls immediately
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0,
                hand_brake=True
            )
        )
        time.sleep(0.5)
        # Create new BasicAgent
        self.agent = BasicAgent(self.vehicle)
        self.agent.ignore_traffic_lights(True)
        self.agent.ignore_stop_signs(True)
        # Attach new camera
        camera_bp = (
            self.world
            .get_blueprint_library()
            .find("sensor.camera.rgb")
        )
        camera_bp.set_attribute(
            "image_size_x",
            str(IM_WIDTH)
        )
        camera_bp.set_attribute(
            "image_size_y",
            str(IM_HEIGHT)
        )
        camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(
                    x=2.5,
                    z=0.7
                )
            ),
            attach_to=self.vehicle
        )
        camera.listen(process_image)
        self.actor_list.append(camera)
        # Attach new collision sensor
        collision_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        collision_sensor.listen(self.process_collision)
        self.actor_list.append(collision_sensor)
        # Give the vehicle a new destination
        destination = random.choice(spawn_points).location
        while destination.distance(self.vehicle.get_location()) < 50:
            destination = random.choice(spawn_points).location

        self.agent.set_destination(destination)
        # Reset driving state
        self.target_speed = 30
        self.stuck_counter = 0
        # Release brake and start normally
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=0.0,
                steer=0.0,
                hand_brake=False
            )
        )
        # Wait for camera frame
        global LatestFrame
        LatestFrame = None
        while LatestFrame is None:
            time.sleep(0.01)
        print("Respawn complete")




        



        