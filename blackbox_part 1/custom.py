"""Custom runs: the user's own query, any supported model, an optional injected fault.

A custom run goes through the same agent adapters, Recorder, fault injection and store as
every other run, so diagnosis, replay and verification work on it unchanged. What is
different is how the outcome is judged, because a free-text query has no built-in answer:

  expected answer given by the user            -> compared with the agent's answer
  query is about facts in the local fact table -> expected answer computed from the table
  fault-injection run, nothing else known      -> a fault-free reference run is recorded first
                                                  and its answer is the expected one
  none of these                                -> "completed": the run counts as a success when
                                                  it ends with a number and no step errored

How the outcome was judged is saved in the run under `custom`, and replays of the run are
judged the same way.
"""
from __future__ import annotations

import os

import inject
import llm as llm_mod
import store
import tasks
import tools
from adapters import FRAMEWORKS, get_agent
from recorder import Recorder

MAX_QUERY = 2000
FAULT_HELP = {
    "empty_search": "A search returns nothing.",
    "wrong_tool_result": "A search returns a fact about a different entity.",
    "wrong_number": "A reader or the calculator returns a wrong number.",
    "skipped_step": "A tool call returns an empty result with no error.",
    "wrong_tool": "The plan sends the first lookup to the calculator instead of search.",
}


class CustomError(ValueError):
    """Bad request for a custom run; the message is shown to the user as is."""


def judge(answer: str, steps: list[dict], expected: float | None = None, tolerance: float = 0.06) -> tuple[str, str]:
    """(outcome, how it was judged)."""
    value = tasks.parse_answer(answer)
    if expected is not None:
        ok = value is not None and abs(value - expected) <= max(tolerance, 0.005 * abs(expected))
        return ("success" if ok else "fail"), "expected"
    clean = value is not None and not any(s.get("error") for s in steps)
    return ("success" if clean else "fail"), "completed"


def expected_from_facts(query: str) -> float | None:
    """Expected answer when the query is one of the question shapes the local fact table can answer."""
    t = tasks.get_task_by_text(query)
    if t:
        return t.expected
    spec = tasks.parse_task(query)
    if spec and all(e in tasks.CORPUS.get(spec.attr, {}) for e in spec.entities) and \
            len(spec.entities) >= (1 if spec.op in ("add", "avg") else 2):
        try:
            return tasks.compute_expected(spec)
        except (ZeroDivisionError, IndexError):
            return None
    return None


def ollama_models(timeout: float = 0.8) -> list[str] | None:
    """Models installed in the local Ollama, or None when Ollama is not reachable."""
    import requests
    host = os.environ.get("BLACKBOX_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:  # noqa: BLE001
        return None


def supported_models() -> dict:
    """What can actually run right now. Hosted providers appear only when their key is set."""
    out = [{"id": m, "group": "Offline mock", "available": True,
            "note": "Rule-based stand-in. Only understands the four built-in question shapes."}
           for m in ("mock-large", "mock-small")]
    installed = ollama_models()
    for m in installed or []:
        out.append({"id": m, "group": "Ollama (local)", "available": True, "note": "Local model."})
    providers = []
    for prefix, (base, key_env) in llm_mod.PROVIDERS.items():
        ready = bool(os.environ.get(key_env)) and bool(base or os.environ.get("BLACKBOX_LLM_BASE_URL"))
        providers.append({"prefix": prefix, "key_env": key_env, "available": ready})
    ready = {p["prefix"] for p in providers if p["available"]}
    with store._conn() as c:   # hosted models that already produced a successful run here
        used = [r[0] for r in c.execute("SELECT DISTINCT model FROM runs WHERE outcome='success' AND model LIKE '%:%'")]
    for m in sorted(used):
        if m.partition(":")[0] in ready:
            out.append({"id": m, "group": "Hosted API", "available": True, "note": "Used successfully before on this machine."})
    return {"models": out, "providers": providers, "ollama_running": installed is not None}


def check_model(model: str, query: str) -> None:
    """Raise CustomError unless this model can run this query right now."""
    model = (model or "").strip()
    if not model:
        raise CustomError("Pick a model.")
    if model.startswith("mock"):
        if model not in ("mock-large", "mock-small"):
            raise CustomError(f"Unknown mock model `{model}`. Use mock-large or mock-small.")
        if tasks.parse_task(query) is None:
            raise CustomError("The mock models only understand the four built-in question shapes (combined population, "
                              "how much taller, how many times larger an area, average river length). "
                              "Pick a real model for a free-form query.")
        return
    prefix, sep, name = model.partition(":")
    if sep and prefix in llm_mod.PROVIDERS:
        base, key_env = llm_mod.PROVIDERS[prefix]
        if not name:
            raise CustomError(f"Add the model name after `{prefix}:`.")
        if not os.environ.get(key_env):
            raise CustomError(f"{key_env} is not set. Set it before starting the dashboard to use `{prefix}:` models.")
        if not base and not os.environ.get("BLACKBOX_LLM_BASE_URL"):
            raise CustomError("BLACKBOX_LLM_BASE_URL is not set, so `api:` models have nowhere to go.")
        return
    installed = ollama_models()
    if installed is None:
        raise CustomError(f"`{model}` would run on Ollama, but Ollama is not reachable on this machine. "
                          "Start Ollama, or use a hosted model such as groq:<model>.")
    name = model.removeprefix("ollama:")
    if name not in installed and f"{name}:latest" not in installed:
        raise CustomError(f"Ollama does not have `{name}` installed. Run: ollama pull {name}")


def check_fault(fault_type: str | None, fault_step: int | None) -> dict:
    if fault_type not in inject.FAULT_TYPES:
        raise CustomError(f"Pick a fault type: {', '.join(inject.FAULT_TYPES)}.")
    if fault_step is not None and not (1 <= int(fault_step) <= 60):
        raise CustomError("Fault step must be between 1 and 60, or left empty for the first step the fault fits.")
    return {"type": fault_type, "step_no": int(fault_step) if fault_step else None, "seed": 1}


def _execute(query, model, framework, fault, expected, tolerance, meta):
    agent = get_agent(framework, llm_mod.get_llm(model))
    rec = Recorder(framework, model.removeprefix("ollama:"), query, None, fault=fault)
    answer = agent.run(query, rec)
    outcome, how = judge(answer, rec.steps, expected, tolerance)
    run = rec.finish(answer, outcome)
    run["custom"] = {**meta, "expected": expected, "tolerance": tolerance, "judged_by": how}
    store.save_run(run)
    return run


def record_custom(query: str, model: str, framework: str = "custom", fault: dict | None = None,
                  expected: float | None = None, tolerance: float = 0.06) -> dict:
    """Run the agent on the user's query and save it. Returns {"run", "reference"}.

    The query is passed to the agent exactly as typed. `fault` None means no fault is injected."""
    query = (query or "").strip()
    if len(query) < 3:
        raise CustomError("Type a query first.")
    if len(query) > MAX_QUERY:
        raise CustomError(f"The query is too long ({len(query)} characters; the limit is {MAX_QUERY}).")
    if framework not in FRAMEWORKS:
        raise CustomError(f"Unknown framework `{framework}`. Supported: {', '.join(FRAMEWORKS)}.")
    check_model(model, query)
    source = "user" if expected is not None else None
    if expected is None:
        expected = expected_from_facts(query)
        source = "fact_table" if expected is not None else None
    reference = None
    if fault and expected is None:
        # nothing to compare a broken run with: record the same query without the fault first
        reference = _execute(query, model, framework, None, None, tolerance,
                             {"mode": "reference", "expected_source": None})
        value = tasks.parse_answer(reference.get("final_answer") or "")
        if reference["outcome"] == "success" and value is not None:
            expected, source = value, "reference_run"
    meta = {"mode": "fault" if fault else "normal", "expected_source": source,
            "reference_run_id": reference["run_id"] if reference else None}
    return {"run": _execute(query, model, framework, fault, expected, tolerance, meta), "reference": reference}


def knowledge() -> list[dict]:
    """What the built-in search tool can look up (it is a local fact table, not the web)."""
    units = {"population": "millions of people", "area": "thousand square kilometers", "height": "meters", "length": "kilometers"}
    return [{"attribute": a, "unit": units.get(a, ""), "entities": sorted(ents)} for a, ents in tasks.CORPUS.items()]


def lookups_missed(run: dict) -> int:
    """Search steps that found nothing without a fault being injected there."""
    hit = (run.get("fault") or {}).get("step_no") if (run.get("fault") or {}).get("applied") else None
    return sum(1 for s in run["steps"] if s.get("error") == "empty_result" and s["step_no"] != hit
               and s.get("output") == tools.NOT_FOUND)
