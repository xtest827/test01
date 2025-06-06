import numpy as np
import random
from collections import defaultdict


class RLAgent:
    def __init__(self, actions, method="q_learning"):
        self.actions = actions
        self.method = method  # "q_learning" or "sarsa"
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_min = 0.1  # Minimum exploration
        self.epsilon_decay = 0.995  # Decay rate per episode
        self.q_table = defaultdict(lambda: [0.0 for _ in range(len(self.actions))])

    def learn(self, state, action, reward, next_state, next_action=None):
        current_q = self.q_table[state][action]

        if self.method == "q_learning":
            target_q = reward + self.discount_factor * max(self.q_table[next_state])
        elif self.method == "sarsa":
            if next_action is None:
                raise ValueError("next_action must be provided for SARSA")
            target_q = reward + self.discount_factor * self.q_table[next_state][next_action]
        else:
            raise ValueError("Unknown method: choose 'q_learning' or 'sarsa'")

        self.q_table[state][action] += self.learning_rate * (target_q - current_q)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            return self.arg_max(state_action)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @staticmethod
    def arg_max(state_action):
        max_value = max(state_action)
        max_indices = [i for i, v in enumerate(state_action) if v == max_value]
        return random.choice(max_indices)


# ===============================
# Example usage (Q-Learning)
# ===============================

env = Env()  # Assuming Env() is defined somewhere
agent = RLAgent(actions=list(env.actions), method="q_learning")  # or method="sarsa"

for episode in range(200):
    state = env.reset()

    if agent.method == "sarsa":
        action = agent.get_action(str(state))

    while True:
        env.render()

        if agent.method == "q_learning":
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)
            agent.learn(str(state), action, reward, str(next_state))
        else:  # SARSA
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(str(next_state))
            agent.learn(str(state), action, reward, str(next_state), next_action)
            action = next_action

        state = next_state
        env.print_value_all(agent.q_table)

        if done:
            break

    agent.decay_epsilon()  # Reduce exploration over time
