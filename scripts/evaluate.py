from app.rdp_env import RDPEnv
from app.drl_agent import choose_server, load_model
import numpy as np

load_model()

servers = [
    {"name": "TS1", "cpu": 60, "active_sessions": 30, "max_sessions": 50},
    {"name": "TS2", "cpu": 50, "active_sessions": 20, "max_sessions": 50},
    {"name": "TS3", "cpu": 55, "active_sessions": 25, "max_sessions": 50}
]

env = RDPEnv(servers)

success = 0
failures = 0

state = env.reset()

for _ in range(200):
    action = choose_server(state)
    next_state, _, _, info = env.step(action)

    if info["failure"]:
        failures += 1
    else:
        success += 1

    state = next_state

print("Success rate:", success / 200)
print("Failures:", failures)