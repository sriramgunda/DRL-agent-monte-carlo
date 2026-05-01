from pathlib import Path
import sys

# Ensure project root is on sys.path so `app` package imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rdp_env import RDPEnv
from app.drl_agent import choose_action, update_q, save_model

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

for episode in range(100):
    print("-" * 20)
    print(f"Episode {episode + 1}/100")
    print("-" * 20)
    state = env.reset()

    for step in range(50):
        print(f"Step {step + 1}/50")
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)

        update_q(state, action, reward, next_state)
        state = next_state

save_model()
print("Training complete")