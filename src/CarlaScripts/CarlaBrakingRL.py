#Alex Eliseev
#Script trains a PPO model how to brake in carla at red lights and accelerate otherwise to reach the destination.
import os
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from TrainingGymEnv import CarlaGymEnvironment

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
    "PPO_Braking"
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
    "PPO_Braking"
    /
    "Checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="TrafficLightAgent"
)


# Create CARLA environment
env = CarlaGymEnvironment()


print("Checking environment...")
check_env(env)
print("Environment OK")


# Recurrent PPO with LSTM memory
model = RecurrentPPO(
    policy="MultiInputLstmPolicy",
    env=env,
    # Discount factor
    gamma=0.99,
    # Learning rte
    learning_rate=0.0003,
    # Steps collected before update
    n_steps=8192,
    # Minibatch size
    batch_size=256,
    # LSTM hidden size
    policy_kwargs=dict(
        lstm_hidden_size=256
    ),
    verbose=1,
    device="cuda"
)

model.learn(
    total_timesteps=500000,
    callback=checkpoint_callback
)

MODEL_SAVE_PATH.mkdir(
    parents=True,
    exist_ok=True
)

model.save(
    MODEL_SAVE_PATH / "TrafficLightBrakeAgent_LSTM"
)

print("Training complete")