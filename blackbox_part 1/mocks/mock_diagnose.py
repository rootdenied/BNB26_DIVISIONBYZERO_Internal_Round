"""Mock of Part 2's POST /diagnose (same contract).  Port 8001.

  uvicorn mocks.mock_diagnose:app --port 8001
"""
from fastapi import FastAPI
from typing import Any

from mocks.heuristics import diagnose

app = FastAPI(title="MOCK diagnose (stand-in for Part 2)")


@app.get("/health")
def health():
    return {"status": "ok", "service": "diagnose", "mock": True}


@app.post("/diagnose")
def post_diagnose(run: dict[str, Any]):
    return diagnose(run)
