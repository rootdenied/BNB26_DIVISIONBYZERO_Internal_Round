"""Fake-failure maker: take a good run and break one step on purpose.

make_failure   - for simulated runs: re-executes the agent with the fault, so
                 later steps really react to it. The broken step is the label.
break_offline  - for runs recorded by Part 1 (or anyone): edits the saved JSON.
                 Later steps are only patched by text substitution, so prefer
                 Part 1's fault flag when it exists.
"""
import copy
import random

from . import sim_agent
from .signals import NUM


def make_failure(clean_run, r, tries=5):
    for t in range(tries):
        spec = copy.deepcopy(clean_run["sim_spec"])
        spec["fault"] = sim_agent.random_fault(spec, r)
        run = sim_agent.execute(spec, f"{clean_run['run_id']}_f{r.randrange(10**6)}")
        if run["outcome"] == "fail" and run["faulty_step"]:
            run["source_run_id"] = clean_run["run_id"]
            return run
    return None


def break_offline(run, fault_type, step_no, r=None):
    """fault_type: wrong_number | wrong_tool_result | empty_search. Returns None if not applicable."""
    r = r or random.Random(0)
    new = copy.deepcopy(run)
    step = next(s for s in new["steps"] if s["step_no"] == step_no)
    old = str(step.get("output") or "")
    swaps = {}
    if fault_type == "empty_search":
        if step.get("kind") != "tool_call":
            return None
        step["output"], step["error"] = "", "empty result"
    else:
        found = NUM.findall(old)
        if not found:
            return None
        target = found[0]
        wrong = sim_agent.fmt(sim_agent.mutate(float(target.replace(",", "")), r))
        step["output"] = old.replace(target, wrong, 1)
        swaps = {target: wrong, target.replace(",", ""): wrong}
    for s in new["steps"]:
        if s["step_no"] in sim_agent.descendants(new, step_no):
            for a, b in swaps.items():
                s["input"] = str(s.get("input") or "").replace(a, b)
                s["output"] = str(s.get("output") or "").replace(a, b)
    new.update(run_id=f"{run['run_id']}_{fault_type}_{step_no}", outcome="fail",
               faulty_step=step_no, fault_type=fault_type, source_run_id=run["run_id"])
    return new
