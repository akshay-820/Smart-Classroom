import os
import yaml
import argparse
import pandas as pd
import mlflow

from sim.classroom_env import SmartClassroomEnv
from agent.qlearning_agent import QLearningAgent


# -----------------------------
# ARGUMENT PARSER
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to config yaml file"
)

args = parser.parse_args()


# -----------------------------
# LOAD CONFIG
# -----------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)


# -----------------------------
# CREATE FOLDERS
# -----------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("policies", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# -----------------------------
# ENVIRONMENT
# -----------------------------
env = SmartClassroomEnv()


# -----------------------------
# AGENT
# -----------------------------
agent = QLearningAgent(
    actions=[0, 1, 2, 3, 4],
    learning_rate=config["learning_rate"],
    discount_factor=config["discount_factor"],
    epsilon=config["epsilon"]
)


# -----------------------------
# MLFLOW SETUP
# -----------------------------
mlflow.set_experiment("Smart_Classroom_Optimization")


# -----------------------------
# TRAINING
# -----------------------------
results = []

with mlflow.start_run(run_name=config["run_name"]):

    # Log hyperparameters
    mlflow.log_params(config)

    episodes = config["episodes"]
    steps_per_episode = config["steps_per_episode"]

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0

        # Epsilon decay
        agent.epsilon = max(
            0.01,
            agent.epsilon * config["epsilon_decay"]
        )

        for step in range(steps_per_episode):

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            agent.update_q_table(
                state,
                action,
                reward,
                next_state
            )

            state = next_state

            total_reward += reward

            if done:
                break

        print(
            f"Episode {episode+1} | "
            f"Reward: {total_reward:.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

        # Store results
        results.append({
            "run_id": config["run_name"],
            "episode": episode + 1,
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "learning_rate": agent.lr,
            "discount_factor": agent.gamma
        })

        # MLflow metrics
        mlflow.log_metric(
            "reward",
            total_reward,
            step=episode
        )

        mlflow.log_metric(
            "epsilon",
            agent.epsilon,
            step=episode
        )

    # -----------------------------
    # SAVE RESULTS CSV
    # -----------------------------
    df = pd.DataFrame(results)

    df.to_csv(
        config["results_path"],
        index=False
    )

    # -----------------------------
    # SAVE POLICY
    # -----------------------------
    agent.save_policy(
        config["policy_path"]
    )

    # -----------------------------
    # LOG ARTIFACTS
    # -----------------------------
    mlflow.log_artifact(config["results_path"])

    mlflow.log_artifact(config["policy_path"])

print("\nTraining completed successfully.")