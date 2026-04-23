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
        cpu = server["cpu"] - completions * 0.2 + np.random.normal(0, 5)
        memory = server["memory"] - completions * 0.5 + np.random.normal(0, 5)

        sessions = sessions + arrivals - completions
        sessions = max(sessions, 0)

        # 3. Resource calculations
        cpu = server["cpu"] + sessions * 1.2 + np.random.normal(0, 5)
        memory = server["memory"] + sessions * 1.5 + np.random.normal(0, 5)
        latency = server["latency"] + cpu * 2 + np.random.normal(0, 20)

        # 4. Failure check (ANY time step)
        if cpu > 85:
            return True, f"CPU overload at minute {t}", t
        if memory > 90:
            return True, f"Memory overload at minute {t}", t
        if sessions > server["max_sessions"]:
            return True, f"Session limit exceeded at minute {t}", t
        if latency > 300:
            return True, f"High latency at minute {t}", t

    return False, "No failure", minutes

def monte_carlo_failure(server, simulations=5000):
    failures = 0

    for _ in range(simulations):
        if simulate_once(server):
            failures += 1

    probability = failures / simulations
    return probability

def monte_carlo_time(server, simulations=3000):
    failures = 0
    failure_times = []
    reasons = []

    for _ in range(simulations):
        failed, reason, failed_time = simulate_over_time(server)

        if failed:
            failures += 1
            failure_times.append(failed_time)
            reasons.append(reason)

    probability = failures / simulations

    return probability, reasons, failure_times

# Example server
server = {
    "name": "TS0",
    "cpu": 10,
    "memory": 25,
    "active_sessions": 15,
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
    "cpu": 18,
    "memory": 75,
    "active_sessions": 5,
    "max_sessions": 50,
    "latency": 120
}

TS3 = {
    "name": "TS3",
    "cpu": 40,
    "memory": 35,
    "active_sessions": 49,
    "max_sessions": 50,
    "latency": 180
}

def evaluate_server(server):
    prob, reasons, failure_times = monte_carlo_time(server)

    return {
        "name": server["name"],
        "failure_probability": prob,
        "risk_level": get_risk_level(prob),
        "top_issue": reasons[0] if reasons else "None"
    }


def get_risk_level(prob):
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


servers = [server,TS1, TS2, TS3]
#servers = os.getenv("SERVERNAMES")

def testing():
    for s in servers:
        prob = monte_carlo_failure(s)
        print(s["name"], prob)


    prob = monte_carlo_time(server)
    print(f"Failure Probability: {prob[0]:.2%}")
    print(f"Failure Reasons: {prob[1][:5]}...")  # Show first 5 reasons
    print(f"Failure Times (minutes): {prob[2][:5]}...")  # Show first 5 failure times

def recommend_server(servers):
    results = []

    for s in servers:
        result = evaluate_server(s)
        results.append(result)
    
        print(f"Server: {result['name']}")
        print(f"  Failure Probability: {result['failure_probability']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Top Issue: {result['top_issue']}")
        print()
    
    # Sort by lowest failure probability
    results.sort(key=lambda x: x["failure_probability"])

    best = results[0]

    return {
        "recommended_server": best["name"],
        "reason": f"Lowest failure probability ({best['failure_probability']:.2f})",
        "all_servers": results
    }

if __name__ == "__main__":
    print("Evaluating servers...")
    recommendation = recommend_server(servers)
    print(f"Recommended Server: {recommendation['recommended_server']}")
    print(f"Reason: {recommendation['reason']}")
    print("All Servers:")
    for server in recommendation['all_servers']:
        print(f"  {server['name']}: {server['failure_probability']:.2%} ({server['risk_level']})")