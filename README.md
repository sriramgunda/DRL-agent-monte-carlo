# AI-Based RDP Routing System

## Run Training
python scripts/train.py

## Run API
uvicorn app.main:app --reload

## Test API
POST /route

{
  "servers": [
    {"name":"TS1", "cpu": 90, "memory": 92, "latency": 250, "active_sessions": 45, "max_sessions": 50},
    {"name":"TS2", "cpu": 50, "memory": 70, "latency": 140, "active_sessions": 30, "max_sessions": 50},
    {"name":"TS3", "cpu": 10, "memory": 20, "latency": 100, "active_sessions": 22, "max_sessions": 50},
    {"name":"TS4", "cpu": 22, "memory": 33, "latency": 120, "active_sessions": 15, "max_sessions": 50},
    {"name":"TS5", "cpu": 34, "memory": 45, "latency": 160, "active_sessions": 29, "max_sessions": 50}
]
}