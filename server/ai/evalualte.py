import os
import pandas as pd
import mlflow

from sim.classroom_env import SmartClassroomEnv
from agent.qlearning_agent import QLearningAgent


# -----------------------------
# CREATE FOLDERS
# -----------------------------
os.makedirs("results", exist_ok=True)


# -----------------------------
# ENVIRONMENT
# -----------------------------
env = SmartClassroomEnv()


# -----------------------------
# LOAD MODEL 1
# -----------------------------
agent_v1 = QLearningAgent(actions=[0, 1, 2, 3, 4])

agent_v1.load_policy("policies/policy_v1.pkl")

agent_v1.epsilon = 0


# -----------------------------
# LOAD MODEL 2
# -----------------------------
agent_v2 = QLearningAgent(actions=[0, 1, 2, 3, 4])

agent_v2.load_policy("policies/policy_v2.pkl")

agent_v2.epsilon = 0


# -----------------------------
# BASELINE POLICY
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
# EVALUATION FUNCTION
# -----------------------------
def evaluate_agent(agent=None, baseline=False):

    rewards = []

    for ep in range(50):

        state = env.reset()

        total_reward = 0

        for step in range(50):

            if baseline:
                action = baseline_policy(env)

            else:
                action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            state = next_state

            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    return rewards


# -----------------------------
# RUN EVALUATIONS
# -----------------------------
baseline_rewards = evaluate_agent(baseline=True)

v1_rewards = evaluate_agent(agent=agent_v1)

v2_rewards = evaluate_agent(agent=agent_v2)


# -----------------------------
# CALCULATE AVERAGES
# -----------------------------
baseline_avg = sum(baseline_rewards) / len(baseline_rewards)

v1_avg = sum(v1_rewards) / len(v1_rewards)

v2_avg = sum(v2_rewards) / len(v2_rewards)


# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n===== FINAL COMPARISON =====")

print(f"Baseline Avg Reward : {baseline_avg:.2f}")

print(f"RL V1 Avg Reward    : {v1_avg:.2f}")

print(f"RL V2 Avg Reward    : {v2_avg:.2f}")


# -----------------------------
# SAVE COMPARISON CSV
# -----------------------------
comparison = pd.DataFrame({
    "Method": [
        "Baseline",
        "RL_V1",
        "RL_V2"
    ],
    "Average Reward": [
        baseline_avg,
        v1_avg,
        v2_avg
    ]
})

comparison.to_csv(
    "results/comparison.csv",
    index=False
)

print("\nComparison saved to results/comparison.csv")


# -----------------------------
# LOG TO MLFLOW
# -----------------------------
mlflow.set_experiment("Smart_Classroom_Optimization")

with mlflow.start_run(run_name="Final_Evaluation"):

    mlflow.log_metric(
        "baseline_avg_reward",
        baseline_avg
    )

    mlflow.log_metric(
        "rl_v1_avg_reward",
        v1_avg
    )

    mlflow.log_metric(
        "rl_v2_avg_reward",
        v2_avg
    )

    mlflow.log_artifact(
        "results/comparison.csv"
    )

print("\nEvaluation completed successfully.")