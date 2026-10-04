"""Replay API (Part 1 -> called by Part 2's verify loop and by the UI).  Port 8000.

  uvicorn replay_api:app --port 8000
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import replay
import runner
import store
import tasks

app = FastAPI(title="Black Box - Replay API (Part 1)", version="1.0")


class ReplayRequest(BaseModel):
    run_id: Optional[str] = None
    step_no: int
    edit: dict[str, Any]
    run: Optional[dict[str, Any]] = Field(None, description="optional inline run (e.g. Part 2 synthetic run)")
    n_runs: int = 3


class BranchRequest(BaseModel):
    run_id: Optional[str] = None
    step_no: int
    model: Optional[str] = Field(None, description="model for the steps from the checkpoint on (default: the run's own)")
    edit: Optional[dict[str, Any]] = Field(None, description="optional edit at the checkpoint step")
    run: Optional[dict[str, Any]] = None
    n_runs: int = 3


class RecordRequest(BaseModel):
    task_id: str
    model: str = "mock-large"
    framework: str = "custom"
    fault: Optional[dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "replay", "runs": len(store.list_runs(limit=100000))}


@app.post("/replay")
def post_replay(req: ReplayRequest):
    try:
        return replay.smart_replay(run_id=req.run_id, step_no=req.step_no, edit=req.edit,
                                   run=req.run, n_runs=req.n_runs)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except (replay.ReplayError, ValueError) as e:
        raise HTTPException(422, str(e))


@app.post("/branch")
def post_branch(req: BranchRequest):
    """Alternative execution from a checkpoint: steps before step_no are restored from the
    record, step_no and everything after it runs again (optionally with another model)."""
    try:
        return replay.branch(run_id=req.run_id, step_no=req.step_no, model=req.model, edit=req.edit,
                             run=req.run, n_runs=req.n_runs)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except (replay.ReplayError, ValueError) as e:
        raise HTTPException(422, str(e))


@app.get("/compare/{a_id}/{b_id}")
def get_compare(a_id: str, b_id: str):
    """Original vs modified trace, step by step."""
    a, b = store.get_run(a_id), store.get_run(b_id)
    if not a or not b:
        raise HTTPException(404, f"run {a_id if not a else b_id} not found")
    return replay.compare(a, b)


@app.get("/runs")
def get_runs(outcome: Optional[str] = None, include_replays: bool = False, limit: int = 200):
    return store.list_runs(outcome=outcome, include_replays=include_replays, limit=limit)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(404, f"run {run_id} not found")
    return r


@app.get("/runs/{run_id}/rewind/{step_no}")
def get_rewind(run_id: str, step_no: int):
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(404, f"run {run_id} not found")
    try:
        return replay.rewind(r, step_no)
    except replay.ReplayError as e:
        raise HTTPException(422, str(e))


@app.post("/record")
def post_record(req: RecordRequest):
    t = tasks.get_task(req.task_id)
    if not t:
        raise HTTPException(404, f"task {req.task_id} not found")
    return runner.record_run(t, req.model, req.framework, fault=req.fault)
