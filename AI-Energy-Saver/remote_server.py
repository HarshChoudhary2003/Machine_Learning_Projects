# remote_server.py
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from datetime import datetime

app = FastAPI()

# In-memory store: {device_id: latest_payload}
DEVICE_STATE: Dict[str, Dict[str, Any]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest")
def ingest_telemetry(payload: Dict[str, Any] = Body(...)):
    """
    Expected payload from agent:
    {
        "device_id": "rohan-laptop",
        "battery_percent": 83,
        "cpu_usage": 34.5,
        "ram_usage": 61.2,
        "num_apps_running": 143,
        "device_temperature": 60,
        ...
    }
    """
    device_id = payload.get("device_id")
    if not device_id:
        return {"status": "error", "message": "device_id is required"}

    payload["timestamp"] = datetime.utcnow().isoformat()
    DEVICE_STATE[device_id] = payload
    return {"status": "ok"}


@app.get("/devices")
def list_devices():
    """Return list of all devices that have reported at least once."""
    return {"devices": list(DEVICE_STATE.keys())}


@app.get("/latest/{device_id}")
def get_latest(device_id: str):
    """Return latest telemetry for a given device."""
    data = DEVICE_STATE.get(device_id)
    if not data:
        return {"status": "error", "message": "No data for this device_id"}
    return {"status": "ok", "data": data}
