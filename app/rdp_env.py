import numpy as np
from app.monte_carlo import monte_carlo_time
from app.utils import compute_reward, risk_cache, get_risk_key


class RDPEnv:
    def __init__(self, servers):
        self.servers = servers

    def reset(self):
        return self._get_state()

    def step(self, action):
        server = self.servers[action]

        server["active_sessions"] += 1

        cpu = server["cpu"] + server["active_sessions"] * 1.2 + np.random.normal(0, 5)
        server["cpu"] = cpu

        memory = server["memory"] + server["active_sessions"] * 1.5 + np.random.normal(0, 5)
        server["memory"] = memory

        latency = server["latency"] + cpu * 2 + np.random.normal(0, 20)
        server["latency"] = latency

        failure = cpu > 85 or memory > 90 or latency > 300 or server["active_sessions"] > server["max_sessions"]

        avg_sessions = np.mean([s["active_sessions"] for s in self.servers])
        risk = self._get_risk(server)

        reward = compute_reward(
            failure,
            cpu,
            memory,
            latency,
            server["active_sessions"],
            avg_sessions,
            risk
        )

        return self._get_state(), reward, False, {"failure": failure}

    def _get_risk(self, server):
        key = get_risk_key(server)

        if key not in risk_cache:
            risk_cache[key] = monte_carlo_time(server)

        return risk_cache[key]

    def _get_state(self):
        state = []

        for s in self.servers:
            
            risk = self._get_risk(s)

            state.extend([
            s["cpu"] / 100,
            s["memory"] / 100,
            s["latency"] / 300,
            s["active_sessions"] / s["max_sessions"],
            risk
        ])

        return np.array(state)