"""Stand-in for Part 2's diagnosis model: simple, honest heuristics (no labels used).

Set MOCK_ORACLE=1 to force the labelled faulty_step to the top (UI demos only).
"""
from __future__ import annotations

import os

import replay
import tasks


def _signals(run: dict) -> dict[int, list[str]]:
    steps = run["steps"]
    n = len(steps) or 1
    by_no = {s["step_no"]: s for s in steps}
    sig: dict[int, list[str]] = {s["step_no"]: [] for s in steps}
    seen_inputs = set()
    for i, s in enumerate(steps):
        k = s["step_no"]
        out = s.get("output") or ""
        if s.get("error") and s["error"] != "empty_result":
            sig[k].append("tool_error")
            for d in s.get("depends_on", []):         # the step that set this up is suspicious too
                if by_no.get(d, {}).get("kind") == "decision":
                    sig[d].append("upstream_of_error")
        if s.get("error") == "empty_result" or (s["kind"] == "tool_call" and not out.strip()):
            sig[k].append("empty_result")
        key = (s["actor"], s["input"])
        if key in seen_inputs:
            sig[k].append("repeated_action")
        seen_inputs.add(key)
        if s["kind"] == "tool_call" and out and i + 1 < len(steps):
            nxt = steps[i + 1]
            v = tasks.parse_number(out)
            nv = tasks.parse_number(nxt.get("output") or "")
            if k in nxt.get("depends_on", []) and v is not None and nv is not None and abs(v - nv) > 1e-9 \
                    and nxt["kind"] == "model_call" and nxt["actor"] == "reader":
                sig[nxt["step_no"]].append("contradiction")
        if (s.get("output") or "").strip().lower().startswith("unknown") or "unknown" in out.lower()[:40]:
            sig[k].append("result_ignored")
        if len(replay.dependents(run, k)) >= max(2, n // 3):
            sig[k].append("many_dependents")
    return sig


WEIGHTS = {"tool_error": 0.45, "upstream_of_error": 0.5, "empty_result": 0.55, "repeated_action": 0.2,
           "contradiction": 0.5, "result_ignored": 0.1, "many_dependents": 0.05}


def diagnose(run: dict, k: int = 3) -> dict:
    sig = _signals(run)
    n = len(run["steps"]) or 1
    scored = []
    for s in run["steps"]:
        no = s["step_no"]
        score = sum(WEIGHTS.get(x, 0.0) for x in sig[no]) + 0.02 * (1 - no / n)
        if s["kind"] == "tool_call" and s["actor"] == "search_agent":
            score += 0.03
        scored.append((score, no))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if os.environ.get("MOCK_ORACLE") == "1" and run.get("faulty_step"):
        fs = run["faulty_step"]
        scored = [(9.0, fs)] + [x for x in scored if x[1] != fs]
        sig.setdefault(fs, []).append("oracle")
    top = scored[:k]
    total = sum(max(s, 0.01) for s, _ in top) or 1
    return {"suspects": [
        {"step_no": no, "confidence": round(min(0.99, max(s, 0.01) / total), 2),
         "evidence": sorted(set(sig.get(no, []))) or ["position"]}
        for s, no in top]}
