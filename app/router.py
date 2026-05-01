import numpy as np
from app.drl_agent import choose_server
from app.monte_carlo import monte_carlo_time
from app.dqn_agent import DQNAgent


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

def score_server(s, risk):
    # normalized metrics
    cpu = s["cpu"] / 100
    mem = s["memory"] / 100
    lat = s["latency"] / 300
    sess = s["active_sessions"] / s["max_sessions"]

    # lower is better
    score = (
        cpu * 0.25 +
        mem * 0.30 +
        lat * 0.25 +
        sess * 0.10 +
        risk * 0.10
    )

    reasons = []
    if cpu > 0.7: reasons.append("High CPU")
    if mem > 0.8: reasons.append("High Memory")
    if lat > 0.6: reasons.append("High Latency")
    if risk > 0.5: reasons.append("High Failure Risk")

    return score, reasons

# ---------------------------
# Build state
# ---------------------------
def build_state(servers):
    state = []
    risks = []

    for s in servers:
        risk = monte_carlo_time(s)
        risks.append(risk)

        state.extend([
            s["cpu"] / 100,
            s["memory"] / 100,
            s["latency"] / 300,
            s["active_sessions"] / s["max_sessions"],
            risk
        ])

    return np.array(state), risks

# ---------------------------
# Confidence estimation
# ---------------------------
def compute_confidence(q_values, action):
    import numpy as np

    # Softmax for probability-like confidence
    exp_q = np.exp(q_values - np.max(q_values))
    probs = exp_q / exp_q.sum()

    return probs[action], probs


def route_request_old(servers):
    state, risks = build_state(servers)
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

# ---------------------------
# Explainability
# ---------------------------
def explain_server(s, risk):
    reasons = []

    if s["cpu"] > 75:
        reasons.append("High CPU")
    if s["memory"] > 80:
        reasons.append("High Memory")
    if s["latency"] > 200:
        reasons.append("High Latency")
    if risk > 0.5:
        reasons.append("High Failure Risk")

    return reasons if reasons else ["Healthy server"]


def route_request_q_learning(servers):
    server_details = []

    for i, s in enumerate(servers):
        risk = monte_carlo_time(s)
        score, reasons = score_server(s, risk)

        server_details.append({
            "index": i,
            "name": s["name"],
            "cpu": s["cpu"],
            "memory": s["memory"],
            "latency": s["latency"],
            "sessions": s["active_sessions"],
            "risk": round(risk, 2),
            "score": round(score, 3),
            "issues": reasons
        })

    # sort by best score
    ranked = sorted(server_details, key=lambda x: x["score"])

    selected = ranked[0]

    return {
        "selected_server": selected["name"],
        "explanation": {
            "reason": "Lowest combined risk score",
            "top_factors": selected["issues"] or ["Healthy server"],
        },
        "ranking": ranked
    }

def route_request_dqn_not_using(servers):
    # Step 1: Build state + risk
    state, risks = build_state(servers)
    action_dim = len(servers)
    # Initialize once
    dqn_agent = DQNAgent(state_dim=len(state), action_dim=action_dim)
    dqn_agent.load()

    # Step 2: Get Q-values + action
    q_values = dqn_agent.get_q_values(state)
    action = int(np.argmax(q_values))

    # Step 3: Compute confidence
    confidence, all_probs = compute_confidence(q_values, action)

    # Step 4: Safety filter
    safe_servers = filter_safe_servers(servers)

    if not safe_servers:
        return {
            "error": "All servers overloaded",
            "action": "REJECT"
        }

    # Step 5: Override if unsafe
    if action not in safe_servers:
        safe_servers_sorted = sorted(
            safe_servers,
            key=lambda i: risks[i]
        )
        action = safe_servers_sorted[0]
        override = True
    else:
        override = False

    selected = servers[action]

    # Step 6: Explainability
    explanation = explain_server(selected, risks[action])

    # Step 7: Ranking score (relative comparison)
    server_scores = []
    for i, s in enumerate(servers):
        score = (
            (s["cpu"]/100)*0.25 +
            (s["memory"]/100)*0.30 +
            (s["latency"]/300)*0.25 +
            (s["active_sessions"]/s["max_sessions"])*0.10 +
            risks[i]*0.10
        )
        server_scores.append(score)

    best_score = min(server_scores)
    worst_score = max(server_scores)

    # Normalize score (0–1, higher = better)
    selected_score = server_scores[action]
    normalized_score = 1 - (
        (selected_score - best_score) / (worst_score - best_score + 1e-6)
    )

    return {
        "selected_server": selected["name"],
        "server_index": action,
        "confidence": round(float(confidence), 3),
        "relative_score": round(float(normalized_score), 3),
        "override_applied": override,

        "risk": round(risks[action], 2),
        "explanation": explanation,

        "all_servers": [
            {
                "name": s["name"],
                "cpu": s["cpu"],
                "memory": s["memory"],
                "latency": s["latency"],
                "sessions": s["active_sessions"],
                "risk": round(risks[i], 2),
                "q_value": round(float(q_values[i]), 3),
                "probability": round(float(all_probs[i]), 3)
            }
            for i, s in enumerate(servers)
        ]
    }


def route_request(servers):
    dqn_agent = DQNAgent(input_dim=5)  # dynamic features per server
    dqn_agent.load()

    best_score = -float("inf")
    best_index = None
    explanations = []

    for i, s in enumerate(servers):

        risk = monte_carlo_time(s)

        # per-server state
        state = [
            s["cpu"] / 100,
            s["memory"] / 100,
            s["latency"] / 300,
            s["active_sessions"] / s["max_sessions"],
            risk
        ]

        score = dqn_agent.predict_score(state)

        explanations.append({
            "name": s["name"],
            "score": round(score, 3),
            "risk": round(risk, 2)
        })

        if score > best_score:
            best_score = score
            best_index = i

    selected = servers[best_index]

    return {
        "selected_server": selected["name"],
        "score": round(best_score, 3),
        "ranking": sorted(explanations, key=lambda x: -x["score"])
    }