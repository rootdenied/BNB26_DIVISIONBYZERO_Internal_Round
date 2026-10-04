"""Recorder: catches every model call / tool call / decision and saves it as a contract step.

The same Recorder powers three modes:
  * record   - normal run (optionally with an injected fault)
  * replay   - re-executes the agent, but REUSES saved outputs for steps that are not
               affected by the edit, and reruns only the dependent ones.
  * branch   - restores every step before a checkpoint from the saved record and runs
               everything from the checkpoint on for real (optionally with another model).

State restoration: a saved answer is restored when the step about to run has the same
actor, kind and input as a saved step that the edit does not affect. Matching is by
content, not by step number, so saved answers are still found when an edit makes the
agent take more or fewer steps than before.

depends_on rule (agreed with Part 2):
  step N depends on (a) the steps the agent explicitly says it used, plus
  (b) any earlier step whose output text (>= 12 chars) appears in N's input, or whose
      numbers (>= 3 chars, commas stripped) appear in N's input.
"""
from __future__ import annotations

import datetime as _dt
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import inject

MIN_TEXT_MATCH = 12  # short words like "unknown" also appear in prompt templates
_NUM_RX = re.compile(r"\d[\d,]*\.?\d*")


def _nums(text: str) -> set[str]:
    out = set()
    for n in _NUM_RX.findall(text or ""):
        n = n.replace(",", "").rstrip(".")
        if len(n) >= 3:
            out.add(n)
    return out


def detect_deps(prev_steps: list[dict], input_text: str) -> set[int]:
    deps, in_nums = set(), _nums(input_text)
    for s in prev_steps:
        out = s.get("output") or ""
        if not out:
            continue
        if len(out) >= MIN_TEXT_MATCH and out in input_text:
            deps.add(s["step_no"])
        elif _nums(out) & in_nums:
            deps.add(s["step_no"])
    return deps


@dataclass
class StepOut:
    step_no: int
    output: str
    error: str | None


@dataclass
class ReplayCtx:
    original: dict
    edit_step: int
    edit: dict                      # {"output": str} | {"input": str} | {"rerun": True}
    dirty: set[int]                 # edit_step + its transitive dependents (original map)
    checkpoint: bool = False        # branch mode: every step from edit_step on is dirty
    orig_steps: dict[int, dict] = field(default_factory=dict)
    by_key: dict[tuple, list[int]] = field(default_factory=dict)
    used: set[int] = field(default_factory=set)

    def __post_init__(self):
        self.orig_steps = {s["step_no"]: s for s in self.original["steps"]}
        for s in self.original["steps"]:
            self.by_key.setdefault((s["actor"], s["kind"], s["input"]), []).append(s["step_no"])

    def restorable(self, s: int, actor: str, kind: str, input: str) -> int | None:
        """Saved step whose answer can be restored for the step about to run, or None."""
        free = [n for n in self.by_key.get((actor, kind, input), ())
                if n not in self.dirty and n not in self.used]
        if not free:
            return None
        return s if s in free else free[0]


class Recorder:
    def __init__(self, framework: str, model: str, task: str, task_id: str | None = None,
                 fault: dict | None = None, replay: ReplayCtx | None = None, auto_deps: bool = True):
        self.framework, self.model, self.task, self.task_id = framework, model, task, task_id
        self.fault = dict(fault) if fault else None
        self.fault_applied = False
        self.replay = replay
        self.auto_deps = auto_deps
        self.steps: list[dict] = []
        self.status: dict[int, str] = {}   # step_no -> fresh | reused | rerun | edited
        self.restored_from: dict[int, int] = {}   # new step_no -> saved step_no it was restored from
        self.timing: list[dict] = []       # per step: when it started (UTC) and how long it took

    # ------------------------------------------------------------------ core
    def step(self, actor: str, kind: str, input: str,
             fn: Callable[[str], tuple[str, int, str | None]], deps=()) -> StepOut:
        """fn(input) -> (output, tokens, error)."""
        s = len(self.steps) + 1
        started, t0 = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"), time.perf_counter()
        rp, status = self.replay, "fresh"
        tokens = 0
        if rp and s == rp.edit_step and "input" in rp.edit:
            input = str(rp.edit["input"])
        src = rp.restorable(s, actor, kind, input) if rp and s != rp.edit_step else None

        if rp and s == rp.edit_step and "output" in rp.edit:
            output, error, status = str(rp.edit["output"]), None, "edited"
        elif rp and s == rp.edit_step:          # edited input or plain retry
            output, tokens, error = fn(input)
            status = "rerun" if rp.checkpoint and "input" not in rp.edit else "edited"
        elif src is not None:
            orig = rp.orig_steps[src]
            output, error, tokens = orig["output"], orig.get("error"), orig.get("tokens", 0)
            status = "reused"
            rp.used.add(src)
            self.restored_from[s] = src
        else:
            output, tokens, error = fn(input)
            status = "rerun" if rp else "fresh"

        if self.fault and not self.fault_applied and status == "fresh" \
                and self.fault.get("step_no") in (None, s) and inject.compatible(self.fault["type"], actor, kind):
            output, error, ok = inject.mutate(self.fault, actor, kind, input, output, error)
            if ok:
                self.fault_applied, self.fault["step_no"] = True, s

        dep_set = {d for d in deps if d is not None and 0 < d < s}
        if self.auto_deps:
            dep_set |= detect_deps(self.steps, input)
        self.steps.append({
            "step_no": s, "actor": actor, "kind": kind, "input": input, "output": output,
            "depends_on": sorted(dep_set), "error": error, "tokens": int(tokens),
        })
        self.status[s] = status
        self.timing.append({"step_no": s, "started_at": started, "ms": round(1000 * (time.perf_counter() - t0), 1)})
        return StepOut(s, output, error)

    # ------------------------------------------------- one-line wrappers for any agent
    def wrap_model(self, llm, actor: str = "agent", kind: str = "model_call"):
        def call(prompt: str, deps=(), json_mode: bool = False) -> str:
            def fn(p):
                r = llm.complete(p, json_mode=json_mode)
                return r.text, r.tokens, r.error
            return self.step(actor, kind, prompt, fn, deps).output
        return call

    def wrap_tool(self, tool_fn: Callable[[str], str], actor: str):
        def call(arg: str, deps=()) -> str:
            def fn(a):
                try:
                    return str(tool_fn(a)), 0, None
                except Exception as e:  # noqa: BLE001
                    return f"ERROR: {e}", 0, getattr(e, "code", type(e).__name__)
            return self.step(actor, "tool_call", arg, fn, deps).output
        return call

    # ------------------------------------------------------------------ finish
    def finish(self, final_answer: str, outcome: str) -> dict:
        run = {
            "run_id": None, "framework": self.framework, "model": self.model,
            "task": self.task, "outcome": outcome, "faulty_step": None, "steps": self.steps,
            # ---- optional, additive fields (Part 2 may ignore) ----
            "task_id": self.task_id, "final_answer": final_answer,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "timing": self.timing,
        }
        if self.fault:
            run["fault"] = {**self.fault, "applied": self.fault_applied}
            if self.fault_applied and outcome == "fail":
                run["faulty_step"] = self.fault["step_no"]
        return run
