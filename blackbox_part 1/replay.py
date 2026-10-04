"""Rewind, dependency map and smart replay.

Smart replay re-executes the agent from the top with a ReplayCtx:
  * steps NOT in the dirty set whose input is unchanged -> saved output reused (no LLM/tool call)
  * the edited step -> edit applied ({"output"}, {"input"} or {"rerun": true})
  * dependents of the edit (or any step whose input changed) -> rerun for real
Each replay runs n_runs times because LLM answers vary.

Branch (alternative execution from a checkpoint): every step before the checkpoint is
restored from the saved record, and every step from the checkpoint on runs for real,
optionally with a different model. Use it to ask "what if a different model had taken
over from here?" without paying for the steps before the checkpoint again.
"""
from __future__ import annotations

import difflib
import hashlib
import json

import llm as llm_mod
import store
import tasks
from adapters import get_agent
from recorder import Recorder, ReplayCtx


class ReplayError(ValueError):
    pass


def dependency_map(run: dict) -> dict[int, list[int]]:
    return {s["step_no"]: list(s.get("depends_on", [])) for s in run["steps"]}


def dependents(run: dict, step_no: int) -> set[int]:
    """All later steps that (transitively) depend on step_no."""
    children: dict[int, set[int]] = {}
    for s, deps in dependency_map(run).items():
        for d in deps:
            children.setdefault(d, set()).add(s)
    seen, stack = set(), [step_no]
    while stack:
        for c in children.get(stack.pop(), ()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def state_hash(steps: list[dict]) -> str:
    """Fingerprint of a list of steps (what was asked, what came back). Two runs whose
    first N steps have the same hash are in the same state after step N."""
    h = hashlib.sha256()
    for s in steps:
        h.update(json.dumps([s["actor"], s["kind"], s["input"], s["output"], s.get("error")]).encode())
    return h.hexdigest()[:12]


def rewind(run: dict, step_no: int) -> dict:
    """State of the run just before step_no: what is frozen, what an edit would rerun,
    and the agent state that a replay restores."""
    steps = {s["step_no"]: s for s in run["steps"]}
    if step_no not in steps:
        raise ReplayError(f"step {step_no} not in run")
    dep = dependents(run, step_no)
    before = [s for s in run["steps"] if s["step_no"] < step_no]
    plan = None
    for s in before:
        if s["kind"] == "decision":
            plan = llm_mod.extract_json(s["output"]) or plan
    return {
        "step": steps[step_no],
        "before": before,
        "will_rerun": sorted(dep),
        "will_reuse": sorted(s for s in steps if s != step_no and s not in dep),
        "state": {
            "checkpoint": step_no,
            "steps_done": len(before),
            "tokens_spent": sum(s["tokens"] for s in before),
            "state_hash": state_hash(before),
            "plan": plan,
            "results_so_far": [{"step_no": s["step_no"], "actor": s["actor"], "output": s["output"],
                                "error": s.get("error")} for s in before if s["kind"] != "decision"],
            "branch_would_rerun": [s for s in steps if s >= step_no],
        },
    }


def _same(x: dict, y: dict) -> bool:
    return all(x.get(k) == y.get(k) for k in ("actor", "kind", "input", "output", "error"))


def compare(a: dict, b: dict) -> dict:
    """Step-by-step comparison of two traces (original a, modified b).

    Steps are aligned on (actor, kind), so the comparison still lines up when the modified
    run has more or fewer steps. Each row is same | changed | added | removed."""
    ka = [(s["actor"], s["kind"]) for s in a["steps"]]
    kb = [(s["actor"], s["kind"]) for s in b["steps"]]
    status_b = (b.get("replay") or {}).get("status", {})
    rows = []

    def row(x, y):
        st = "removed" if y is None else "added" if x is None else "same" if _same(x, y) else "changed"
        ref = y or x
        return {"a_step": x and x["step_no"], "b_step": y and y["step_no"], "actor": ref["actor"],
                "kind": ref["kind"], "status": st,
                "input_changed": bool(x and y and x["input"] != y["input"]),
                "output_changed": bool(x and y and x["output"] != y["output"]),
                "error_changed": bool(x and y and x.get("error") != y.get("error")),
                "a_input": x and x["input"], "b_input": y and y["input"],
                "a_output": x and x["output"], "b_output": y and y["output"],
                "a_error": x and x.get("error"), "b_error": y and y.get("error"),
                "replay_status": status_b.get(str(y["step_no"])) if y else None}

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes():
        if tag in ("equal", "replace"):
            n = min(i2 - i1, j2 - j1)
            rows += [row(a["steps"][i1 + k], b["steps"][j1 + k]) for k in range(n)]
            rows += [row(x, None) for x in a["steps"][i1 + n:i2]]
            rows += [row(None, y) for y in b["steps"][j1 + n:j2]]
        elif tag == "delete":
            rows += [row(x, None) for x in a["steps"][i1:i2]]
        else:
            rows += [row(None, y) for y in b["steps"][j1:j2]]
    count = {k: sum(r["status"] == k for r in rows) for k in ("same", "changed", "added", "removed")}
    first = next((r for r in rows if r["status"] != "same"), None)
    ta, tb = sum(s["tokens"] for s in a["steps"]), sum(s["tokens"] for s in b["steps"])
    return {
        "a_run_id": a.get("run_id"), "b_run_id": b.get("run_id"),
        "summary": {**count,
                    "first_divergence": first and (first["b_step"] or first["a_step"]),
                    "outcome": [a.get("outcome"), b.get("outcome")],
                    "outcome_changed": a.get("outcome") != b.get("outcome"),
                    "final_answer": [a.get("final_answer"), b.get("final_answer")],
                    "steps": [len(a["steps"]), len(b["steps"])], "tokens": [ta, tb],
                    "model": [a.get("model"), b.get("model")]},
        "steps": rows,
    }


def _validate_edit(edit: dict) -> dict:
    if not isinstance(edit, dict):
        raise ReplayError("edit must be an object")
    keys = [k for k in ("output", "input", "rerun") if k in edit]
    if len(keys) != 1:
        raise ReplayError('edit needs exactly one of "output", "input", "rerun"')
    if keys[0] == "rerun" and not edit["rerun"]:
        raise ReplayError('"rerun" must be true')
    return {keys[0]: edit[keys[0]]}


def _resolve_task(run: dict) -> tasks.Task | None:
    """The built-in task of a run, or None for a custom run (judged by its own `custom` record)."""
    t = tasks.get_task(run.get("task_id") or "") or tasks.get_task_by_text(run["task"])
    if t is None and not run.get("custom"):
        raise ReplayError("unknown task: cannot judge the replayed outcome")
    return t


def _judge(run: dict, task: tasks.Task | None, answer: str, steps: list[dict]) -> str:
    if task is not None:
        return "success" if tasks.check_answer(task, answer) else "fail"
    import custom
    c = run["custom"]
    return custom.judge(answer, steps, c.get("expected"), c.get("tolerance", 0.06))[0]


def replay_once(run: dict, step_no: int, edit: dict, llm=None, model: str | None = None,
                branch: bool = False) -> dict:
    task = _resolve_task(run)
    later = {s["step_no"] for s in run["steps"] if s["step_no"] >= step_no}
    dirty = later if branch else {step_no} | dependents(run, step_no)
    ctx = ReplayCtx(original=run, edit_step=step_no, edit=edit, dirty=dirty, checkpoint=branch)
    use_model = model or run["model"]
    agent = get_agent(run["framework"], llm or llm_mod.get_llm(use_model))
    rec = Recorder(run["framework"], use_model, run["task"], task.task_id if task else None, replay=ctx)
    answer = agent.run(run["task"], rec)
    new = rec.finish(answer, _judge(run, task, answer, rec.steps))
    if run.get("custom"):
        new["custom"] = run["custom"]
    reused = [s for s, st in rec.status.items() if st == "reused"]
    rerun = [s for s, st in rec.status.items() if st in ("rerun", "edited", "fresh")]
    full = sum(s["tokens"] for s in new["steps"])
    saved = sum(s["tokens"] for s in new["steps"] if s["step_no"] in reused)
    # did the replay really start from the recorded state? every step before the checkpoint
    # must come back identical and from the saved record, not from a fresh call
    prefix = [s for s in run["steps"] if s["step_no"] < step_no]
    got = {s["step_no"]: s for s in new["steps"]}
    bad = [s["step_no"] for s in prefix
           if s["step_no"] not in got or not _same(s, got[s["step_no"]]) or rec.status.get(s["step_no"]) != "reused"]
    new["parent_run_id"] = run.get("run_id")
    new["replay"] = {"parent_run_id": run.get("run_id"), "edited_step": step_no, "edit": edit,
                     "mode": "branch" if branch else "smart", "model": use_model,
                     "predicted_rerun": sorted(dirty), "reused_steps": reused, "rerun_steps": rerun,
                     "status": {str(k): v for k, v in rec.status.items()},
                     "restored_from": {str(k): v for k, v in rec.restored_from.items()},
                     "restore": {"checkpoint": step_no, "prefix_steps": len(prefix),
                                 "restored": len(prefix) - len(bad), "exact": not bad,
                                 "mismatched_steps": bad, "state_hash": state_hash(prefix),
                                 "replayed_hash": state_hash([s for s in new["steps"] if s["step_no"] < step_no])}}
    new["_stats"] = {"steps_rerun": len(rerun), "steps_reused": len(reused),
                     "tokens_saved_pct": (100.0 * saved / full) if full else 0.0}
    return new


def smart_replay(run_id: str | None = None, step_no: int | None = None, edit: dict | None = None,
                 run: dict | None = None, n_runs: int = 3, save: bool = True, llm=None, db=None,
                 model: str | None = None, branch: bool = False) -> dict:
    """Implements the POST /replay contract (and POST /branch when branch=True)."""
    if run is None:
        if not run_id:
            raise ReplayError("need run_id or run")
        run = store.get_run(run_id, db=db)
        if run is None:
            raise LookupError(f"run {run_id} not found")
    if step_no is None or step_no not in {s["step_no"] for s in run["steps"]}:
        raise ReplayError(f"step_no {step_no} not in run")
    edit = _validate_edit({"rerun": True} if branch and edit is None else edit)
    n_runs = max(1, min(int(n_runs), 10))
    if not run.get("run_id"):
        run["run_id"] = run_id

    replays = [replay_once(run, step_no, edit, llm=llm, model=model, branch=branch) for _ in range(n_runs)]
    stats = [r.pop("_stats") for r in replays]
    if save:
        for r in replays:
            store.save_run(r, db=db)
    wins = [r for r in replays if r["outcome"] == "success"]
    success_rate = len(wins) / n_runs
    rep = wins[0] if wins else replays[0]
    majority = "success" if success_rate > 0.5 else "fail"
    avg = lambda k: sum(s[k] for s in stats) / n_runs  # noqa: E731
    return {
        "new_run": rep,
        "outcome_flipped": majority != run["outcome"],
        "steps_rerun": round(avg("steps_rerun")),
        "steps_reused": round(avg("steps_reused")),
        "tokens_saved_pct": round(avg("tokens_saved_pct")),
        # ---- additive fields ----
        "original_run_id": run.get("run_id"),
        "n_runs": n_runs,
        "success_rate": round(success_rate, 3),
        "replays": [{"run_id": r["run_id"], "outcome": r["outcome"], **s} for r, s in zip(replays, stats)],
        "mode": "branch" if branch else "smart",
        "model": model or run["model"],
        "state_restored": all(r["replay"]["restore"]["exact"] for r in replays),
        "restore": rep["replay"]["restore"],
        "comparison": compare(run, rep),
    }


def branch(run_id: str | None = None, step_no: int | None = None, model: str | None = None,
           edit: dict | None = None, **kw) -> dict:
    """Alternative execution from a checkpoint: restore everything before step_no, then run
    step_no and all later steps for real, optionally with another model or an edit."""
    return smart_replay(run_id=run_id, step_no=step_no, edit=edit, model=model, branch=True, **kw)
