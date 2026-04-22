import numpy as np
import random

def simulate_once(server):
    # Simulate random changes
    cpu = server["cpu"] + np.random.normal(5, 10)
    memory = server["memory"] + np.random.normal(5, 8)
    sessions = server["active_sessions"] + random.randint(0, 5)
    latency = server["latency"] + np.random.normal(20, 30)

    # Failure conditions
    if cpu > 85:
        return True
    if memory > 90:
        return True
    if sessions > server["max_sessions"]:
        return True
    if latency > 300:
        return True

    return False


def monte_carlo_failure(server, simulations=5000):
    failures = 0

    for _ in range(simulations):
        if simulate_once(server):
            failures += 1

    probability = failures / simulations
    return probability


# Example server
server = {
    "name": "TS1",
    "cpu": 70,
    "memory": 75,
    "active_sessions": 45,
    "max_sessions": 50,
    "latency": 150
}

TS1 = {
    "name": "TS1",
    "cpu": 70,
    "memory": 75,
    "active_sessions": 45,
    "max_sessions": 50,
    "latency": 150
}

TS2 = {
    "name": "TS2",
    "cpu": 80,
    "memory": 55,
    "active_sessions": 5,
    "max_sessions": 50,
    "latency": 250
}

TS3 = {
    "name": "TS3",
    "cpu": 90,
    "memory": 85,
    "active_sessions": 49,
    "max_sessions": 50,
    "latency": 450
}

prob = monte_carlo_failure(server)
print(f"Failure Probability: {prob:.2f}")

servers = [TS1, TS2, TS3]

for s in servers:
    prob = monte_carlo_failure(s)
    print(s["name"], prob)