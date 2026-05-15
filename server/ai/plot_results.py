import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow


# -----------------------------
# CREATE PLOTS FOLDER
# -----------------------------
os.makedirs("plots", exist_ok=True)


# -----------------------------
# LOAD TRAINING RESULTS
# -----------------------------
v1_df = pd.read_csv("results/results_v1.csv")

v2_df = pd.read_csv("results/results_v2.csv")

comparison_df = pd.read_csv("results/comparison.csv")


# -----------------------------
# PLOT 1 - MODEL V1 TRAINING
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    v1_df["episode"],
    v1_df["reward"]
)

plt.title("RL Model V1 Training Rewards")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.grid(True)

plt.savefig("plots/reward_plot_v1.png")

plt.close()


# -----------------------------
# PLOT 2 - MODEL V2 TRAINING
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    v2_df["episode"],
    v2_df["reward"]
)

plt.title("RL Model V2 Training Rewards")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.grid(True)

plt.savefig("plots/reward_plot_v2.png")

plt.close()


# -----------------------------
# PLOT 3 - FINAL COMPARISON
# -----------------------------
plt.figure(figsize=(8, 5))

plt.bar(
    comparison_df["Method"],
    comparison_df["Average Reward"]
)

plt.title("Baseline vs RL Comparison")

plt.ylabel("Average Reward")

plt.savefig("plots/comparison_plot.png")

plt.close()

print("\nPlots generated successfully.")


# -----------------------------
# LOG PLOTS TO MLFLOW
# -----------------------------
mlflow.set_experiment("Smart_Classroom_Optimization")

with mlflow.start_run(run_name="Plots"):

    mlflow.log_artifact(
        "plots/reward_plot_v1.png"
    )

    mlflow.log_artifact(
        "plots/reward_plot_v2.png"
    )

    mlflow.log_artifact(
        "plots/comparison_plot.png"
    )

print("\nPlots logged to MLflow.")