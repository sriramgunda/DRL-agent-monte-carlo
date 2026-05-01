import numpy as np
import random

class RDPEnv:

    def __init__(self, servers):
        self.servers = servers

    def reset(self):
        return self._get_state()

    def step(self, action):
        server = self.servers[action]

        # Simulate session allocation
        server["active_sessions"] += 1

        # Recompute metrics
        cpu = 20 + server["active_sessions"] * 1.2 + np.random.normal(0, 5)
        server["cpu"] = cpu

        # Failure condition
        failure = cpu > 85 or server["active_sessions"] > server["max_sessions"]

        # Reward
        reward = 10 if not failure else -20
        reward -= cpu * 0.05

        next_state = self._get_state()

        done = False

        return next_state, reward, done

    def _get_state(self):
        state = []
        for s in self.servers:
            state.extend([s["cpu"], s["active_sessions"]])
        return np.array(state)


class DRLAgent:

    def __init__(self, action_size):
        self.action_size = action_size
        self.q_table = {}

    def get_action(self, state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_key])

    def update_q_value(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        best_next_q = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += alpha * (reward + gamma * best_next_q - self.q_table[state_key][action])