from pathlib import Path
from pyexpat import model
import sys

# Ensure project root is on sys.path so `app` package imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rdp_env import RDPEnv
from app.drl_agent import choose_action, update_q, save_model
from app.dqn_agent import DQNAgent
from app.monte_carlo import monte_carlo_time
import numpy as np

servers = [
    {
        "name": "TS0",
        "cpu": 10,
        "memory": 25,
        "active_sessions": 15,
        "max_sessions": 50,
        "latency": 150
    },
    {
        "name": "TS1",
        "cpu": 70,
        "memory": 75,
        "active_sessions": 45,
        "max_sessions": 50,
        "latency": 150
    },
    {
        "name": "TS2",
        "cpu": 18,
        "memory": 75,
        "active_sessions": 5,
        "max_sessions": 50,
        "latency": 120
    },
    {
        "name": "TS3",
        "cpu": 40,
        "memory": 35,
        "active_sessions": 49,
        "max_sessions": 50,
        "latency": 180
    }
]

env = RDPEnv(servers)

def build_server_state(server):
    risk = monte_carlo_time(server)

    return [
        server["cpu"] / 100,
        server["memory"] / 100,
        server["latency"] / 300,
        server["active_sessions"] / server["max_sessions"],
        risk
    ]

def q_learning_train():

    for episode in range(100):
        print("-" * 20)
        print(f"Episode: {episode + 1}/100")
        print("-" * 20)
        state = env.reset()

        for step in range(50):
            print(f"Step: {step + 1}/50")
            action = choose_action(state)
            next_state, reward, done, _ = env.step(action)

            update_q(state, action, reward, next_state)
            state = next_state

def double_q_learning_train():
    #state = env.reset()
    #agent = DQNAgent(state_dim=len(state), action_dim=len(servers))
    agent = DQNAgent(input_dim=5)  # 5 features per server

    for episode in range(100):
        print("-" * 20)
        print(f"Episode: {episode + 1}/100")
        print("-" * 20)
        state = env.reset()

        for step in range(50):
            print(f"Step: {step + 1}/50")
            #action = agent.act(state)
            scores = []
            states = []

            for s in servers:
                state = build_server_state(s)  # get current state for each server
                states.append(state)

                score = agent.predict_score(state)
                scores.append(score)

            # Choose best server
            action = np.argmax(scores)

            # Take action in env
            next_state_env, reward, _, info = env.step(action)

            # Store ONLY selected server state
            state = states[action]

            # Build next state for same server
            next_state = build_server_state(servers[action])

            agent.memory.append((state, reward, next_state))

            agent.train()
    
    agent.save(path="data/dqn_model.pth")


q_learning_train()
save_model()
print("Q-learning training complete, model saved.")

double_q_learning_train()
print("DQN training complete, model saved")

print("Training complete")