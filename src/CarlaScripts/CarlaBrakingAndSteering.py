#Alex Eliseev
#Script trains a PPO model how to brake in carla at a single red light intersection
import os
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from TrainingGymEnvTurning import TrainingGymEnvTurning

import torch


print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

MODEL_DIR = (
    PROJECT_ROOT
    / "MastersMRPEndToEndSelfDrivingRover"
    / "Models"
    / "ReinforcementLearningModelsCarla"
    / "SAC_Braking_And_Steering"
)

CHECKPOINT_DIR = (
    MODEL_DIR
    / "Checkpoints"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "SAC_TrafficLight_LaneAgent"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# CHECKPOINT CALLBACK
# --------------------------------------------------

checkpoint_callback = CheckpointCallback(
    save_freq=35000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="SAC_TrafficLight_LaneAgent"
)


# --------------------------------------------------
# CREATE CARLA ENVIRONMENT
# --------------------------------------------------

env = TrainingGymEnvTurning()


print("Checking environment...")

check_env(
    env,
    warn=True
)

print("Environment OK")


# --------------------------------------------------
# CREATE SAC MODEL
# --------------------------------------------------

model = SAC(

    policy="MultiInputPolicy",

    env=env,

    # Discount factor
    gamma=0.99,

    # Learning rate
    learning_rate=0.0003,

    # Number of transitions stored
    buffer_size=500000,

    # Collect some experience before training starts
    learning_starts=10000,

    # Number of samples per training update
    batch_size=256,

    # Soft target-network update amount
    tau=0.005,

    # Train after every environment step
    train_freq=1,

    # One gradient update per training step
    gradient_steps=1,

    # Automatically tune SAC entropy/exploration
    ent_coef="auto",

    verbose=1,

    device="cuda",

    tensorboard_log="./tensorboard/SAC_Braking_And_Steering/"
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback
)


# --------------------------------------------------
# SAVE FINAL MODEL
# --------------------------------------------------

model.save(
    FINAL_MODEL_PATH
)

print("Training complete")