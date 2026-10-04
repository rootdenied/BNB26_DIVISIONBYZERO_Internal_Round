"""Stand-in for Part 1's POST /replay so Part B can test alone (simulated runs only).

    uvicorn part2.mock_replay_api:app --port 8000
"""
import os

from fastapi import Body, FastAPI, HTTPException

from . import sim_agent
from .data import DATA, read_jsonl

app = FastAPI(title="Mock replay")
_runs = {}


@app.post("/replay")
def replay(payload: dict = Body(...)):
    if not _runs:
        for name in ("failed.jsonl", "clean.jsonl"):
            _runs.update({r["run_id"]: r for r in read_jsonl(os.path.join(DATA, name))})
    run = _runs.get(payload.get("run_id"))
    if not run:
        raise HTTPException(404, "unknown run_id")
    return sim_agent.replay(run, int(payload["step_no"]), payload["edit"]["output"])
