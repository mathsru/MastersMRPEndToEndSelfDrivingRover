# Plot rewards per episode from the training log

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_FILE_LOCATION = (
    PROJECT_ROOT
    / "MastersMRPEndToEndSelfDrivingRover"
    / "Models"
    / "ReinforcementLearningModelsCarla"
    / "PPO_Braking_1_light"
    / "Farming_The_Red.csv"
)

# Read CSV
df = pd.read_csv(LOG_FILE_LOCATION)

# If episode numbering restarts (e.g. after resuming training),
# create a continuous episode index for plotting.
df["TrainingEpisode"] = range(1, len(df) + 1)

WINDOW = 20

# Moving average
df["Reward_MA"] = (
    df["Reward"]
    .rolling(window=WINDOW, min_periods=1)
    .mean()
)

plt.figure(figsize=(16, 7))

# Plot reward for every episode
plt.plot(
    df["TrainingEpisode"],
    df["Reward"],
    linewidth=1,
    alpha=0.5,
    label="Episode Reward"
)

# Plot moving average
plt.plot(
    df["TrainingEpisode"],
    df["Reward_MA"],
    linewidth=3,
    color="orange",
    label=f"{WINDOW}-Episode Moving Average"
)

plt.title("Reward per Episode")
plt.xlabel("Training Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()