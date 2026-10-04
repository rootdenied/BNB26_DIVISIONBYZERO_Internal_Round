"""POST /diagnose: takes a run, returns the top 3 suspect steps.
POST /explain: the same suspects plus a plain-language explanation of the failure.

    uvicorn part2.diagnose_api:app --port 8001
"""
from fastapi import Body, FastAPI, HTTPException

from . import explain as explainer
from . import model

app = FastAPI(title="Black Box diagnosis")
_booster = None


def booster():
    global _booster
    if _booster is None:
        _booster = model.load()
    return _booster


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/diagnose")
def diagnose(payload: dict = Body(...)):
    run = payload.get("run", payload)  # accepts the run itself or {"run": {...}}
    if not run.get("steps"):
        raise HTTPException(400, "run has no steps")
    return {"run_id": run.get("run_id"), "suspects": model.diagnose(booster(), run)}


@app.post("/explain")
def explain(payload: dict = Body(...)):
    """Body: the run, or {"run": {...}, "llm": "groq:<model>"}. Without `llm` (and without the
    BLACKBOX_EXPLAIN_LLM environment variable) the explanation is built from the model's evidence."""
    run = payload.get("run", payload)
    if not run.get("steps"):
        raise HTTPException(400, "run has no steps")
    suspects = model.diagnose(booster(), run)
    llm = payload.get("llm") if "run" in payload else None
    return {"run_id": run.get("run_id"), "suspects": suspects,
            "explanation": explainer.explain(run, suspects, llm)}
