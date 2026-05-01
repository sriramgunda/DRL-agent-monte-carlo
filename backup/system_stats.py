import numpy as np
import random

def system_metrics(server):
    # Simulate random changes
    cpu = server["cpu"] + np.random.normal(5, 10)
    memory = server["memory"] + np.random.normal(5, 8)
    sessions = server["active_sessions"] + random.randint(0, 5)
    latency = server["latency"] + np.random.normal(20, 30)

    return {
        "cpu": cpu,
        "memory": memory,
        "active_sessions": sessions,
        "latency": latency
    }

# Example server
server = {
    "name": "TS1",
    "cpu": 70,
    "memory": 75,
    "active_sessions": 45,
    "max_sessions": 50,
    "latency": 150
}

if __name__ == "__main__":
    metrics = system_metrics(server)
    print(f"Simulated Metrics for {server['name']}:")
    print(f"CPU Usage: {metrics['cpu']:.2f}%")
    print(f"Memory Usage: {metrics['memory']:.2f}%")
    print(f"Active Sessions: {metrics['active_sessions']}")
    print(f"Latency: {metrics['latency']:.2f} ms")