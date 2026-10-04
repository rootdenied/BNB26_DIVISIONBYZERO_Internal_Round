"""Fault injection: make the agent ACTUALLY run with one broken step.

Pass fault={"step_no": 4, "type": "empty_search"} to a run. With step_no None the fault
hits the first compatible step. The broken step becomes the run's faulty_step label.
"""
from __future__ import annotations

import json
import random

import tasks
import tools

FAULT_TYPES = ["empty_search", "wrong_tool_result", "wrong_number", "skipped_step", "wrong_tool"]


def compatible(ftype: str, actor: str, kind: str) -> bool:
    if ftype in ("empty_search", "wrong_tool_result"):
        return kind == "tool_call" and actor == "search_agent"
    if ftype == "wrong_number":
        return (kind == "model_call" and actor == "reader") or (kind == "tool_call" and actor == "calculator")
    if ftype == "skipped_step":
        return kind == "tool_call"
    if ftype == "wrong_tool":
        return kind == "decision" and actor == "planner"
    return False


def mutate(fault: dict, actor: str, kind: str, inp: str, out: str, err):
    """Return (output, error, applied)."""
    ftype, seed = fault["type"], fault.get("seed", 0)
    rng = random.Random(seed)
    if ftype == "empty_search":
        return tools.NOT_FOUND, "empty_result", True
    if ftype == "wrong_tool_result":
        return tools.distractor(inp, seed), None, True
    if ftype == "wrong_number":
        v = tasks.parse_number(out)
        if v is None:
            return out, err, False
        factor = rng.choice([10.0, 0.1, 1.37, 0.71])
        new = tasks.fmt_value(round(v * factor, 2))
        return out.replace(tasks.fmt_value(v), new, 1) if tasks.fmt_value(v) in out else new, None, True
    if ftype == "skipped_step":
        return "", None, True
    if ftype == "wrong_tool":
        plan = None
        try:
            plan = json.loads(out[out.find("{"): out.rfind("}") + 1])
        except Exception:
            pass
        if not plan or not plan.get("steps"):
            return out, err, False
        plan["steps"][0]["tool"] = "calculator"
        return json.dumps(plan), None, True
    return out, err, False


def eligible_steps(run: dict, ftype: str) -> list[int]:
    return [s["step_no"] for s in run["steps"] if compatible(ftype, s["actor"], s["kind"])]


def plan_faults(clean_run: dict, rng: random.Random, types=None) -> list[dict]:
    """One fault per type at a random eligible step of a clean run."""
    out = []
    for t in types or FAULT_TYPES:
        steps = eligible_steps(clean_run, t)
        if steps:
            out.append({"step_no": rng.choice(steps), "type": t, "seed": rng.randint(0, 9999)})
    return out
