from sim.classroom_env import SmartClassroomEnv
from agent.qlearning_agent import QLearningAgent

import pandas as pd

env = SmartClassroomEnv()

# -----------------------------
# Load trained RL policy
# -----------------------------

agent = QLearningAgent(actions=[0, 1, 2, 3, 4])

agent.load_policy("policies/policy_v2.pkl")

# Disable exploration during evaluation
agent.epsilon = 0

# -----------------------------
# Baseline controller
# -----------------------------

def baseline_policy(env):

    # Empty classroom
    if env.students == 0:
        return 0

    # High temperature
    if env.temperature > 30:
        return 2

    # Turn lights ON if needed
    if env.light_on == 0:
        return 1

    return 0

# -----------------------------
# Evaluate baseline
# -----------------------------

baseline_rewards = []

for ep in range(50):

    state = env.reset()

    total_reward = 0

    for step in range(50):

        action = baseline_policy(env)

        next_state, reward, done = env.step(action)

        total_reward += reward

    baseline_rewards.append(total_reward)

# -----------------------------
# Evaluate RL policy
# -----------------------------

rl_rewards = []

for ep in range(50):

    state = env.reset()

    total_reward = 0

    for step in range(50):

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        state = next_state

        total_reward += reward

    rl_rewards.append(total_reward)

# -----------------------------
# Final comparison
# -----------------------------

baseline_avg = sum(baseline_rewards) / len(baseline_rewards)

rl_avg = sum(rl_rewards) / len(rl_rewards)

print("\n===== FINAL COMPARISON =====")

print(f"Baseline Avg Reward: {baseline_avg:.2f}")

print(f"RL Avg Reward: {rl_avg:.2f}")

if rl_avg > baseline_avg:
    print("\nRL performed better than baseline.")
else:
    print("\nBaseline performed better in this run.")

comparison = pd.DataFrame({
    "Method": ["Baseline", "RL"],
    "Average Reward": [baseline_avg, rl_avg]
})

comparison.to_csv(
    "results/comparison.csv",
    index=False
)

print("\nComparison saved to results/comparison.csv")