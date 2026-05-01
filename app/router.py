import numpy as np
from app.drl_agent import choose_server
from app.monte_carlo import monte_carlo_time

def filter_safe_servers(servers):
    safe = []

    for i, s in enumerate(servers):
        if (
            s["cpu"] < 85 and
            s["memory"] < 90 and
            s["latency"] < 300 and
            s["active_sessions"] < s["max_sessions"]
        ):
            safe.append(i)

    return safe

def build_state(servers):
    state = []

    for s in servers:
        risk = monte_carlo_time(s)
        state.extend([
            s["cpu"] / 100,
            s["memory"] / 100,
            s["latency"] / 300,
            s["active_sessions"] / s["max_sessions"],
            risk
        ])

    return np.array(state)


def route_request(servers):
    state = build_state(servers)
    # Step 1: Get safe servers
    safe_servers = filter_safe_servers(servers)

    if not safe_servers:
        return {
            "error": "All servers overloaded",
            "action": "REJECT"
        }

    # Step 2: DRL decision
    action = choose_server(state)
    print("-" * 30)
    print("STATE:", state)
    print("ACTION:", action)
    print("-" * 30)

    # Step 3: Safety override (IMPORTANT)
    if action not in safe_servers:
        # fallback: choose safest server
        safe_servers_sorted = sorted(
            safe_servers,
            key=lambda i: monte_carlo_time(servers[i])
        )

        action = safe_servers_sorted[0]

    return {
        "selected_server": servers[action]["name"],
        "server_index": action,
        "safe_servers": safe_servers
    }