# AI-Based RDP Routing System

## Run Training
python scripts/train.py

## Run API
uvicorn app.main:app --reload

## Test API
POST /route

{
  "servers" = [
    {"name":"TS1", "cpu": 90, "memory": 92, "latency": 250, "active_sessions": 45, "max_sessions": 50},
    {"name":"TS2", "cpu": 50, "memory": 60, "latency": 100, "active_sessions": 20, "max_sessions": 50}
]
}