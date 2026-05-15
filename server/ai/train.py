from sim.classroom_env import SmartClassroomEnv
from agent.qlearning_agent import QLearningAgent

import pandas as pd

env = SmartClassroomEnv()

agent = QLearningAgent(
    actions=[0, 1, 2, 3, 4],
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=1.0
)

episodes = 1000

results = []

for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    # Slower epsilon decay
    agent.epsilon = max(0.01, agent.epsilon * 0.99)

    for step in range(50):

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

    print(
        f"Episode {episode+1} | "
        f"Reward: {total_reward:.2f} | "
        f"Epsilon: {agent.epsilon:.3f}"
    )

    # Experiment tracking
    results.append({
        "run_id": "qlearning_v2",
        "episode": episode + 1,
        "reward": total_reward,
        "epsilon": agent.epsilon,
        "learning_rate": agent.lr,
        "discount_factor": agent.gamma
    })

# Save training results
df = pd.DataFrame(results)

df.to_csv(
    "results/training_results.csv",
    index=False
)

# Save learned policy
agent.save_policy(
    "policies/policy_v2.pkl"
)

print("\nTraining completed successfully.")