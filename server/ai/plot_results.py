import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Training Results Plot
# -----------------------------

df = pd.read_csv("results/training_results.csv")

plt.figure(figsize=(10, 5))

plt.plot(df["episode"], df["reward"])

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward vs Episodes")

plt.grid(True)

plt.savefig("plots/reward_plot.png")

print("Reward plot saved.")

# -----------------------------
# Comparison Plot
# -----------------------------

comparison = pd.read_csv("results/comparison.csv")

plt.figure(figsize=(6, 5))

plt.bar(
    comparison["Method"],
    comparison["Average Reward"]
)

plt.ylabel("Average Reward")
plt.title("Baseline vs RL")

plt.savefig("plots/comparison_plot.png")

print("Comparison plot saved.")