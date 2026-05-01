from fastapi import FastAPI
from app.router import route_request
from app.drl_agent import load_model

app = FastAPI()

load_model()

@app.post("/route")
def route(data: dict):
    servers = data["servers"]
    return route_request(servers)