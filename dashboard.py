import streamlit as st
import requests
import random

st.title("AI RDP Routing Dashboard")

# Simulated servers
servers = [
    {
        "name": f"TS{i}",
        "cpu": random.randint(40, 90),
        "memory": random.randint(50, 95),
        "latency": random.randint(80, 300),
        "active_sessions": random.randint(10, 50),
        "max_sessions": 50
    }
    for i in range(1, 4)
]

st.subheader("Current Server Metrics")
st.write(servers)

# Call API
response = requests.post("http://localhost:8000/route", json={"servers": servers})
data = response.json()

st.subheader("Routing Decision")
st.success(f"Selected Server: {data['selected_server']}")

st.subheader("Explanation")
st.write(data["explanation"])

st.subheader("Server Ranking")
st.table(data["ranking"])