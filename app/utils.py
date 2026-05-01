risk_cache = {}

def get_risk_key(server):
    return (round(server["cpu"], 1), server["active_sessions"])


def compute_reward(failure, cpu, memory, latency, sessions, avg_sessions, risk):
    reward = 0

    if failure:
        reward -= 40   # stronger punishment
    else:
        reward += 5    # reduce success reward

    # heavy penalties for bad infra
    reward -= (cpu / 100) * 10
    reward -= (memory / 100) * 12
    reward -= (latency / 300) * 15

    # load balancing
    reward -= abs(sessions - avg_sessions) * 1.5

    # future risk penalty
    reward -= risk * 15

    return reward