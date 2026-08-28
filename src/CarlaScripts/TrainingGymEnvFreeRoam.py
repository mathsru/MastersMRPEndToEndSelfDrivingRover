#Alex Eliseev
#This script runs training episodes of RL inside Carla on world 5, spawnning the car at spawn 201 and having it travel forwards towards the intersection ahead and respawning back at 201 upon crossing the itnersection
#My yolov8 model is good at correctly detecting red/green on this traffic light, the model is simply intended to learn to break in the case that it is a red.
#Similar overall structure of GYM environment + BPO model to the other experiment, just simpler.
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
#Initla setup, connecting to carla and creating the camera and setting up the yolo model view
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
class TrainingGymEnvFreeRoam(gym.Env):
    def __init__(self):
        import random
        super().__init__()
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town05")
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        for _ in range(50):
            self.world.tick()
        self.actor_list = []
        self.warmup_time = 5.0
        self.action = 0
        self.episode_start_time = None
        #Car will have 2 choices of actions that it can make, reduce speed to 0 or increase speed to 30
        # 0 = reduce speed to 0
        # 1 = increase speed to 30
        self.action_space = spaces.Discrete(2)
        self.episode_reward = 0
        self.episode_number = 1
        #Logging episodes ran and rewards over time
        self.log_file = (
            PROJECT_ROOT
            / "Models"
            / "ReinforcementLearningModelsCarla"
            / "PPO_Braking_4_Light"
            / "training_log.csv"
        )
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("Episode,Reward,NumberOfBrakesOnARed,Reason\n")
        #Small if to keep writing from the last episode in the log file, assuming training it continuing
        if self.log_file.exists():
            import pandas as pd
            df = pd.read_csv(self.log_file)
            if len(df) > 0:
                self.episode_number = int(df["Episode"].iloc[-1]) + 1
            else:
                self.episode_number = 1
        else:
            self.episode_number = 1
        #Observation space (what is being fed into the model as input features)
        #Model knows the cars current speed, if red light is detected and the camera view
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(128, 128, 3),
                dtype=np.uint8
            ),
            "red_light": spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "current_car_speed": spaces.Box(
                low=0,
                high=60,
                shape=(1,),
                dtype=np.float32
            )
        })
        #Setting traffic lights
        self.traffic_lights = self.world.get_actors().filter("traffic.traffic_light*")

        #Always starting with a green for 30 sec, spawn very close to the light
        self.light_timer = 0
        self.light_phase = 2             
        self.current_cycle_duration = 15.0

        for light in self.traffic_lights:
            light.set_state(carla.TrafficLightState.Green)

        for light in self.traffic_lights:
            light.freeze(True)
        self.vehicle = None
        self.end_location = None
        self.red_counter = 0 #Red light detected counter
        self.stopped_on_red = 0
        self.red_light_memory = 0
        self.previous_speed = 0
        self.braking_on_a_red = 0
        self.target_speed = 0

    #Function to start a new training episode, car spawns at spawn point 201, goes towards intersection light, respawns after passing the light new episode begins
    def reset(self,seed=None,options=None):
        self.episode_reward = 0
        self.red_counter = 0
        self.stopped_on_red = 0
        self.action = 0
        self.previous_speed = 0
        self.braking_on_a_red = 0
        global LatestFrame
        LatestFrame = None
        super().reset(seed=seed)
        self.cleanup()
        self.traffic_lights = self.world.get_actors().filter("traffic.traffic_light*")

        self.light_timer = 0
        self.light_phase = 2
        self.current_cycle_duration = 15.0

        for light in self.traffic_lights:
            light.set_state(carla.TrafficLightState.Green)

        #Setting spawn point
        # Get all CARLA spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        # Pick a random starting spawn
        start_spawn = random.choice(spawn_points)

        #SPawning the car and selecting destination
        blueprint = self.world.get_blueprint_library().filter("model3")[0]
        self.vehicle = self.world.spawn_actor(blueprint,start_spawn)
        self.actor_list.append(self.vehicle)
        

        #Having the default Carla agent control the steering
        self.agent = BasicAgent(self.vehicle)
        self.agent.set_target_speed(35)
        self.choose_new_destination()

        #Spawnning camera
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x",str(IM_WIDTH))
        camera_bp.set_attribute("image_size_y",str(IM_HEIGHT))
        camera = self.world.spawn_actor(camera_bp,carla.Transform(carla.Location(x=2.5,z=0.7)),attach_to=self.vehicle)
        camera.listen(process_image)
        self.actor_list.append(camera)

        #Wait for camera
        while LatestFrame is None:
            self.world.tick()
        self.episode_start_time = time.time()
        observation = self.get_state()
        return observation, {}

    #New step function, only options for the model are to either 0, do nothing or 1, hit the brakes
    #New step function, only options for the model are to either 0, do nothing or 1, hit the brakes
    def step(self, action):
        if time.time() - self.episode_start_time > 300:
            print("Episode timeout")
            observation = self.get_state()

            reward = 100 #Any reward is fine, car reached the end
            self.episode_reward += reward

            with open(self.log_file, "a") as f:
                f.write(
                    f"{self.episode_number},"
                    f"{self.episode_reward:.2f},"
                    f"{self.braking_on_a_red},"
                    f"Timeout\n"
                )

            self.episode_number += 1
            return observation, reward, True, False, {}
        terminated = False
        truncated = False
        #Check if car has reached the next spawn point
        if self.vehicle.get_location().distance(self.end_location) < 20:
            print("Reached destination - choosing new destination")
            self.choose_new_destination()

        # DQN chooses speed
        if action == 0:
            self.target_speed = 35

        elif action == 1:
            self.target_speed = 0

        # CARLA BasicAgent handles steering/navigation
        self.agent.set_target_speed(self.target_speed)

        control = self.agent.run_step()

        self.vehicle.apply_control(control)
        self.update_traffic_lights()
        self.world.tick()

        observation = self.get_state()
        reward = self.calculate_reward(observation)

        self.episode_reward += reward

        if self.episode_reward < -5000:
            print("Episode failed")

            observation = self.get_state()

            if self.braking_on_a_red == 0:
                reason = "Stood Still"
            else:
                reason = "Ran Red"

            reward = -750

            with open(self.log_file, "a") as f:
                f.write(
                    f"{self.episode_number},"
                    f"{self.episode_reward:.2f},"
                    f"{self.braking_on_a_red},"
                    f"{reason}\n"
                )

            self.episode_number += 1

            return observation, reward, True, False, {}

        return observation, reward, terminated, truncated, {}

    #Function returns the current state of the car, so its speed, if red light is detected and the camera view of the road ahead
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

        #Making sure Yolo model cannot pick up opposing traffic lights
        cutoff_x = IM_WIDTH * 0.225
        cutoff_y = IM_HEIGHT * 0.225
        min_x = cutoff_x
        max_x = IM_WIDTH - cutoff_x
        min_y = cutoff_y
        max_y = IM_HEIGHT - cutoff_y

        for box in results.boxes:
            cls = int(box.cls)
            confidence = float(box.conf)

            # Only care about red lights
            if cls != 0:
                continue

            if confidence < 0.2:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)

            # Ignore detections outside camera center region
            if not (
                min_x <= center_x <= max_x
                and
                min_y <= center_y <= max_y
            ):
                continue

            red_light = 1
            break
        if self.red_light_memory == 0:
            self.stopped_on_red = 0

        if red_light == 1: #Incrementing redlight counter because a red was detected
            self.red_counter +=1
            self.red_light_memory = 20
        elif self.red_light_memory > 0: #NExt 5 frames after red was detected are also considered a red, this redcues noise and frames with failed/no detection, the stream can have gaps going red, red, red, no detection, no detectction, red, red, red
            self.red_counter += 0.1
            red_light = 1
            self.red_light_memory -= 1
        else:
            self.red_counter = 0

        return {
            "image": r1_frame,
            "red_light": np.array([red_light], dtype=np.float32),
            "current_car_speed": np.array([velocity],dtype=np.float32)
        }

    #Function calculates reward based off of the current state, the model should be rewarded for stopping/moving slowly when a red light is detected, is progressively punished more and more
    def calculate_reward(self,observation):
        speed = observation["current_car_speed"][0]
        red_light = observation["red_light"][0]
        reward = 0
        speed_drop = self.previous_speed - speed
        self.previous_speed = speed

        #Car gets rewarded for moving towards its destination, for now faster = better
        if red_light == 1 and speed < 1: #Rewarding for stopping during a red
            reward += 7
            if self.stopped_on_red == 0:
                reward +=800 #Big reward for fully stopping on red, to offset high speed penalty
            self.stopped_on_red = 1
        elif red_light == 1 and speed < 5:
            reward += 2
        elif red_light == 1 and speed > 5: #Punishing for driving through a red
            reward -= speed * 3
        else:
            if speed > 3:
                reward += speed * 0.15 #Rewarding for driving/moving forwards
            else:
                    reward -= 7

        #print(reward)
        #print(f"Speed: {speed:.2f} km/h | Reward: {reward}")
        #print(self.is_driving_straight())
        print(self.episode_reward)
        return float(reward)

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

    def update_traffic_lights(self):

        self.light_timer += 0.05

        # GREEN
        if self.light_phase == 0:
            if self.light_timer >= self.current_cycle_duration:
                self.light_timer = 0
                self.light_phase = 2

                # Random red duration
                self.current_cycle_duration = random.choice([10.0, 15.0, 20.0])

                for light in self.traffic_lights:
                    light.set_state(carla.TrafficLightState.Red)

        # RED
        elif self.light_phase == 2:
            if self.light_timer >= self.current_cycle_duration:
                self.light_timer = 0
                self.light_phase = 0

                # Random green duration
                self.current_cycle_duration = random.choice([10.0, 15.0, 20.0])

                for light in self.traffic_lights:
                    light.set_state(carla.TrafficLightState.Green)

    #Function sets a new destination for the car
    def choose_new_destination(self):
        spawn_points = self.world.get_map().get_spawn_points()

        current_location = self.vehicle.get_location()

        destination = random.choice(spawn_points).location

        # Make sure the destination isn't too close to the car
        while destination.distance(current_location) < 50:
            destination = random.choice(spawn_points).location

        self.end_location = destination
        self.agent.set_destination(destination)
        self.target_speed = 30
        



        