#This script tests trained car agent models
from stable_baselines3 import DQN
from TrainingGymEnvFourLight import TrainingGymEnvFourLight
from pathlib import Path

# Create environment
env = TrainingGymEnvFourLight()

# Load trained DQN model
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "MastersMRPEndToEndSelfDrivingRover"
    / "Models"
    / "ReinforcementLearningModelsCarla"
    / "PPO_Braking_4_light"
    / "Checkpoints"
    / "DQN_FourTrafficLightBrakeAgent"
)

print(MODEL_PATH)

model = DQN.load(
    MODEL_PATH,
    env=env,
    device="cuda"
)

# Start an episode
observation, info = env.reset()

while True:

    # Ask the neural network what to do
    action, _ = model.predict(
        observation,
        deterministic=True
    )

    # Execute the action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished.")
        observation, info = env.reset()