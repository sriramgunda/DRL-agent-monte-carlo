import numpy as np
import random

from system_stats import system_metrics
from dotenv import load_dotenv
import os
load_dotenv()

def simulate_once(server):
    # Simulate random changes
    metrics = system_metrics(server)
    cpu = metrics["cpu"]
    memory = metrics["memory"]
    sessions = metrics["active_sessions"]
    latency = metrics["latency"]

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

def simulate_over_time(server, minutes=10):
    sessions = server["active_sessions"]

    for t in range(minutes):

        # 1. User arrivals
        arrivals = np.random.poisson(3)

        # 2. Session completions
        completions = np.random.binomial(sessions, 0.1)

        sessions = sessions + arrivals - completions
        sessions = max(sessions, 0)

        # 3. Resource calculations
        cpu = 20 + sessions * 1.2 + np.random.normal(0, 5)
        memory = 30 + sessions * 1.5 + np.random.normal(0, 5)
        latency = 50 + cpu * 2 + np.random.normal(0, 20)

        # 4. Failure check (ANY time step)
        if cpu > 85:
            return True, f"CPU overload at minute {t}"
        if memory > 90:
            return True, f"Memory overload at minute {t}"
        if sessions > server["max_sessions"]:
            return True, f"Session limit exceeded at minute {t}"
        if latency > 300:
            return True, f"High latency at minute {t}"

    return False, "No failure"

def monte_carlo_failure(server, simulations=5000):
    failures = 0

    for _ in range(simulations):
        if simulate_once(server):
            failures += 1

    probability = failures / simulations
    return probability

def monte_carlo_time(server, simulations=3000):
    failures = 0
    reasons = []

    for _ in range(simulations):
        failed, reason = simulate_over_time(server)

        if failed:
            failures += 1
            reasons.append(reason)

    probability = failures / simulations

    return probability, reasons

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

prob = monte_carlo_time(server)
print(f"Failure Probability: {prob[0]:.2%}")

servers = [TS1, TS2, TS3]
#servers = os.getenv("SERVERNAMES")

for s in servers:
    prob = monte_carlo_failure(s)
    print(s["name"], prob)