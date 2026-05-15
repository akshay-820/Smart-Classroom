import random
import pickle

class QLearningAgent:

    def __init__(self,
                 actions,
                 learning_rate=0.1,
                 discount_factor=0.9,
                 epsilon=0.2):

        self.q_table = {}

        self.actions = actions

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_q_value(self, state, action):

        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):

        # Exploration
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)

        # Exploitation
        q_values = [
            self.get_q_value(state, a)
            for a in self.actions
        ]

        max_q = max(q_values)

        best_actions = [
            a for a, q in zip(self.actions, q_values)
            if q == max_q
        ]

        return random.choice(best_actions)

    def update_q_table(self,
                       state,
                       action,
                       reward,
                       next_state):

        current_q = self.get_q_value(state, action)

        next_q_values = [
            self.get_q_value(next_state, a)
            for a in self.actions
        ]

        max_next_q = max(next_q_values)

        new_q = current_q + self.lr * (
            reward +
            self.gamma * max_next_q -
            current_q
        )

        self.q_table[(state, action)] = new_q

    def save_policy(self, filename):

        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_policy(self, filename):

        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)