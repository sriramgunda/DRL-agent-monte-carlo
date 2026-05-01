import numpy as np
import pickle

q_table = {}

def get_q(state, action):
    return q_table.get((tuple(state), action), 0)


def update_q(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    max_next = max([get_q(next_state, a) for a in range(3)])
    old = get_q(state, action)

    new = old + alpha * (reward + gamma * max_next - old)
    q_table[(tuple(state), action)] = new


def choose_action(state, epsilon=0.2):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 3)

    q_vals = [get_q(state, a) for a in range(3)]
    return int(np.argmax(q_vals))


def choose_server(state):
    q_vals = [get_q(state, a) for a in range(3)]
    return int(np.argmax(q_vals))


def save_model(path="data/q_table.pkl"):
    with open(path, "wb") as f:
        pickle.dump(q_table, f)


def load_model(path="data/q_table.pkl"):
    global q_table
    with open(path, "rb") as f:
        q_table = pickle.load(f)