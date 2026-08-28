#This script tests trained car agent models
from stable_baselines3 import PPO
from TrainingGymEnvSingleLight import TrainingGymEnvSingleLight
from stable_baselines3 import DQN
from TrainingGymEnvSingleLight import TrainingGymEnvSingleLight
from pathlib import Path

# Create environment
env = TrainingGymEnvSingleLight()

# Model path
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "MastersMRPEndToEndSelfDrivingRover"
    / "Models"
    / "ReinforcementLearningModelsCarla"
    / "PPO_Braking_1_light"
    / "Checkpoints"
    / "TrafficLightAgent_560000_steps"
)

print(MODEL_PATH)

# Load trained DQN
model = DQN.load(
    MODEL_PATH,
    env=env
)

# Start an episode
observation, info = env.reset()

while True:

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished.")
        observation, info = env.reset()