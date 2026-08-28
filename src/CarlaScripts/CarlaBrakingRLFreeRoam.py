#Alex Eliseev
#Script trains a PPO model how to brake in carla at a single red light intersection
import os
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from TrainingGymEnvFreeRoam import TrainingGymEnvFreeRoam
from stable_baselines3 import DQN

import torch


print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_SAVE_PATH = (
    PROJECT_ROOT
    /
    "MastersMRPEndToEndSelfDrivingRover"
    /
    "Models"
    /
    "ReinforcementLearningModelsCarla"
    /
    "DQN_Free_Roam_Only_Gas"
    /
    "Checkpoints"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    /
    "MastersMRPEndToEndSelfDrivingRover"
    /
    "Models"
    /
    "ReinforcementLearningModelsCarla"
    /
    "DQN_Free_Roam_Only_Gas"
    /
    "Checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


checkpoint_callback = CheckpointCallback(
    save_freq=35000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="TrafficLightAgent"
)


# Create CARLA environment
env = TrainingGymEnvFreeRoam()


print("Checking environment...")
check_env(env)
print("Environment OK")


# Recurrent PPO with LSTM memory
model = DQN(
    policy="MultiInputPolicy",
    env=env,

    # Discount factor
    gamma=0.99,

    # Learning rate
    learning_rate=0.0001,

    # Number of experiences stored
    buffer_size=100000,

    # Start learning after collecting this many experiences
    learning_starts=5000,

    # How many samples from replay buffer per update
    batch_size=128,

    # Update target network every X environment steps
    target_update_interval=5000,

    # Exploration
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,

    verbose=1,
    tensorboard_log="./tensorboard/"
)
"""MODEL_PATH = (
    MODEL_SAVE_PATH
    / "DQN_FourTrafficLightBrakeAgent"
)

model = DQN.load(
    MODEL_PATH,
    env=env,
    device="cuda"
)"""

model.learn(
    total_timesteps=700000,
    reset_num_timesteps=False,
    callback=checkpoint_callback
)

MODEL_SAVE_PATH.mkdir(
    parents=True,
    exist_ok=True
)

model.save(
    MODEL_SAVE_PATH / "DQN_Free_Roam_Only_Speed_Control"
)
print("Training complete")