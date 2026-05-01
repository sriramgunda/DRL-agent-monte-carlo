import numpy as np

def simulate_over_time(server, minutes=5):
    sessions = server["active_sessions"]

    for t in range(minutes):
        arrivals = np.random.poisson(3)
        completions = np.random.binomial(sessions, 0.1)

        sessions = max(0, sessions + arrivals - completions)

        cpu = server["cpu"] + sessions * 1.2 + np.random.normal(0, 5)
        memory = server["memory"] + sessions * 1.8 + np.random.normal(0, 5)
        latency = server["latency"] + cpu * 1.5 + np.random.normal(0, 20)

        failure_prob = 0

        if cpu > 80:
            failure_prob += (cpu - 80) / 20

        if memory > 80:
            failure_prob += (memory - 80) / 20

        if latency > 200:
            failure_prob += (latency - 200) / 100

        if sessions > server["max_sessions"] * 0.9:
            failure_prob += 0.5

        if np.random.rand() < min(failure_prob, 1.0):
            return True

    return False


def monte_carlo_time(server, simulations=300):
    failures = 0

    for _ in range(simulations):
        if simulate_over_time(server):
            failures += 1

    return failures / simulations