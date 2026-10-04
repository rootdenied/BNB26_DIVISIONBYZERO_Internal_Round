"""HTTP clients for the two cross-team seams, with offline fallbacks so Part 1 can be
tested without Part 2 running.

  DIAGNOSE_URL  (default http://localhost:8001)  -> Part 2's /diagnose
  REPLAY_URL    (default http://localhost:8000)  -> Part 1's own /replay API
"""
from __future__ import annotations

import os

import requests

DIAGNOSE_URL = os.environ.get("DIAGNOSE_URL", "http://localhost:8001").rstrip("/")
REPLAY_URL = os.environ.get("REPLAY_URL", "http://localhost:8000").rstrip("/")


def _clean(run: dict) -> dict:
    return {k: v for k, v in run.items() if not k.startswith("_")}


def diagnose(run: dict, url: str | None = None, timeout: int = 30) -> tuple[dict, str]:
    """Returns (response, source). Source tells the UI whether Part 2 or the mock answered."""
    url = (url or DIAGNOSE_URL).rstrip("/")
    try:
        r = requests.post(f"{url}/diagnose", json=_clean(run), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        from contract.validate import validate_diagnose
        validate_diagnose(data)
        return data, f"diagnose API ({url})"
    except Exception as e:  # noqa: BLE001
        from mocks.heuristics import diagnose as local
        return local(run), f"built-in mock (diagnose API unreachable: {type(e).__name__})"


def explain(run: dict, llm: str | None = None, url: str | None = None, timeout: int = 320) -> dict:
    """Plain-language explanation of a failed run from Part 2's POST /explain.
    Returns the explanation dict, or {"error": ...} when Part 2 is not reachable."""
    url = (url or DIAGNOSE_URL).rstrip("/")
    body = {"run": _clean(run)}
    if llm:
        body["llm"] = llm
    try:
        r = requests.post(f"{url}/explain", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["explanation"]
    except Exception as e:  # noqa: BLE001
        return {"error": f"explanation needs Part 2's API at {url} ({type(e).__name__})"}


def replay(run_id: str, step_no: int, edit: dict | None, n_runs: int = 3, run: dict | None = None,
           url: str | None = None, timeout: int = 600, branch: bool = False,
           model: str | None = None) -> tuple[dict, str]:
    """Smart replay (POST /replay) or, with branch=True, alternative execution from the
    checkpoint (POST /branch, optionally with another model)."""
    url = (url or REPLAY_URL).rstrip("/")
    body = {"run_id": run_id, "step_no": step_no, "n_runs": n_runs}
    if edit is not None:
        body["edit"] = edit
    if branch and model:
        body["model"] = model
    if run is not None:
        body["run"] = _clean(run)
    path = "branch" if branch else "replay"
    try:
        r = requests.post(f"{url}/{path}", json=body, timeout=timeout)
    except requests.ConnectionError:
        import replay as rp
        return rp.smart_replay(run_id=run_id, step_no=step_no, edit=edit, run=run, n_runs=n_runs,
                               branch=branch, model=model if branch else None), \
            "in-process (replay API not running)"
    if r.status_code >= 400:
        raise RuntimeError(f"/{path} {r.status_code}: {r.text}")
    return r.json(), f"replay API ({url})"
