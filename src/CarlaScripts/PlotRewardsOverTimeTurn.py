# Script plots rewards over time for the RL agent learning from the training log

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_FILE_LOCATION = (
    PROJECT_ROOT
    / "MastersMRPEndToEndSelfDrivingRover"
    / "Models"
    / "ReinforcementLearningModelsCarla"
    / "SAC_Braking_And_Steering"
    / "Training_Log.csv"
)

# Read CSV
df = pd.read_csv(LOG_FILE_LOCATION)

WINDOW = 20

# Moving averages
df["Reward_MA"] = (
    df["Reward"]
    .rolling(window=WINDOW, min_periods=1)
    .mean()
)

df["Brakes_MA"] = (
    df["NumberOfBrakesOnARed"]
    .rolling(window=WINDOW, min_periods=1)
    .mean()
)

print(df["Reason"].value_counts())

# ------------------------
# Reward Plot
# ------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    df["Episode"],
    df["Reward"],
    alpha=0.3,
    label="Reward"
)

plt.plot(
    df["Episode"],
    df["Reward_MA"],
    linewidth=3,
    label=f"{WINDOW}-Episode Moving Average"
)

plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ------------------------
# Brake Actions Plot
# ------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    df["Episode"],
    df["NumberOfBrakesOnARed"],
    alpha=0.3,
    label="Brake Actions"
)

plt.plot(
    df["Episode"],
    df["Brakes_MA"],
    linewidth=3,
    label=f"{WINDOW}-Episode Moving Average"
)

plt.title("Brake Actions During Red Lights")
plt.xlabel("Episode")
plt.ylabel("Number of Brake Actions")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()