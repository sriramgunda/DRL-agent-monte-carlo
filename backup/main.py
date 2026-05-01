from fastapi import FastAPI
from simulator import recommend_server

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    servers = data["servers"]

    decision = recommend_server(servers)

    return decision