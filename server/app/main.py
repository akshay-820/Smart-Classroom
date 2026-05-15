from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.simulation_service import SimulationService

app = FastAPI(title="Smart Classroom API")
simulation = SimulationService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def get_status():
    return simulation.get_status()


@app.post("/api/reset")
def reset():
    return simulation.reset()


@app.post("/api/step")
def step():
    return simulation.step()
