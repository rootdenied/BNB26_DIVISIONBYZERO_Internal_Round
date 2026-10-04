"""A simulated agent that writes runs in the shared contract format.

Part B uses this to get training data without waiting for Part 1. It answers
"look up some numbers and combine them" tasks from a toy knowledge base, in the
style of three frameworks and three models. Every downstream step is computed
from the actual outputs of the steps before it, so an injected fault (or a
replay edit) really propagates through the run.
"""
import random
import re

ENTITIES = ["France", "Spain", "Italy", "Poland", "Kenya", "Peru", "Chile",
            "Nepal", "Ghana", "Cuba", "Laos", "Fiji", "Oman", "Mali"]
METRICS = ["population", "land area in sq km", "annual rainfall in mm",
           "number of airports"]
FRAMEWORKS = ["custom", "langgraph", "crewai"]
MODEL_NAMES = ["gpt-4o-mini", "llama3", "mistral"]
FAULT_TYPES = ["wrong_tool_result", "empty_search", "wrong_number",
               "skipped_step", "wrong_tool"]

STYLES = {
    "custom": dict(planner="planner", searcher="search_tool", extractor="extractor",
                   calc="calculator", answer="answerer",
                   router=False, delegate=False, reviewer=False),
    "langgraph": dict(planner="planner_node", searcher="tools_node", extractor="agent_node",
                      calc="tools_node", answer="agent_node",
                      router=True, delegate=False, reviewer=False),
    "crewai": dict(planner="manager", searcher="researcher", extractor="researcher",
                   calc="analyst", answer="writer",
                   router=False, delegate=True, reviewer=True),
}
MODELS = {
    "gpt-4o-mini": dict(retry=0.06, redundant=0.05, verbose=0.15),
    "llama3": dict(retry=0.15, redundant=0.12, verbose=0.50),
    "mistral": dict(retry=0.10, redundant=0.08, verbose=0.85),
}
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def nums(text):
    out = []
    for m in NUM.findall(str(text or "")):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def fmt(x):
    x = round(float(x), 2)
    return str(int(x)) if x == int(x) else str(x)


def kb_value(metric, entity):
    return random.Random(f"kb|{metric}|{entity}").randint(1_000, 9_000_000)


def compute(op, vals):
    if op == "difference":
        return vals[0] - vals[1]
    if op == "average":
        return round(sum(vals) / len(vals), 2)
    return sum(vals)


def expression(op, vals):
    v = [fmt(x) for x in vals]
    if op == "difference":
        return f"{v[0]} - {v[1]}"
    if op == "average":
        return f"({' + '.join(v)}) / {len(v)}"
    return " + ".join(v)


def mutate(v, r):
    """Return a plausible but wrong version of a number."""
    v = float(v)
    choices = [v * 10, v + r.randint(1, 9) * 10 ** r.randint(1, 4), v - r.randint(1, 9) * 100]
    if abs(v) >= 100:
        choices.append(v // 10)
    new = r.choice(choices)
    return new if new != v else v + 1000


def task_text(spec):
    names = spec["entities"]
    listed = ", ".join(names[:-1]) + " and " + names[-1]
    if spec["op"] == "difference":
        return f"How much larger is the {spec['metric']} of {names[0]} than {names[1]}?"
    if spec["op"] == "average":
        return f"What is the average {spec['metric']} of {listed}?"
    return f"What is the total {spec['metric']} of {listed}?"


def truth(spec):
    return compute(spec["op"], [kb_value(spec["metric"], e) for e in spec["entities"]])


def make_spec(seed, framework, model):
    r = random.Random(f"spec|{seed}")
    k = r.choice([2, 2, 3, 3, 4])
    op = r.choice(["sum", "difference", "average"]) if k == 2 else r.choice(["sum", "average"])
    return dict(seed=seed, framework=framework, model=model, metric=r.choice(METRICS),
                entities=r.sample(ENTITIES, k), op=op, fault=None)


def random_fault(spec, r):
    ftype = r.choice(FAULT_TYPES)
    fault = dict(type=ftype, lookup=r.randrange(len(spec["entities"])))
    if ftype == "wrong_number":
        fault["target"] = r.choice(["extract", "extract", "answer"])
    return fault


def execute(spec, run_id, override=None):
    """Run the simulated agent. `override` maps step_no -> edited output (for replay)."""
    override = {int(k): v for k, v in (override or {}).items()}
    st, prof = STYLES[spec["framework"]], MODELS[spec["model"]]
    fault = spec.get("fault") or {}
    ftype, flook = fault.get("type"), fault.get("lookup")
    metric, ents, op, seed = spec["metric"], spec["entities"], spec["op"], spec["seed"]
    steps, faulty = [], [None]

    def rng(label):
        return random.Random(f"{seed}|{label}")

    def say(label, short, long):
        return long if rng(label).random() < prof["verbose"] else short

    def add(actor, kind, inp, out, deps, error=None, is_fault=False):
        n = len(steps) + 1
        if is_fault:
            faulty[0] = n
        if n in override:
            out, error = str(override[n]), None
        tokens = 0 if kind == "tool_call" else (len(str(inp)) + len(str(out))) // 4
        steps.append(dict(step_no=n, actor=actor, kind=kind, input=inp, output=out,
                          depends_on=sorted(set(deps)), error=error, tokens=tokens))
        return n

    def out(n):
        return steps[n - 1]["output"]

    task = task_text(spec)
    plan = "Plan: " + "; ".join(f"look up {metric} of {e}" for e in ents)
    plan += f"; then compute the {op}; then answer."
    p = add(st["planner"], "model_call", task, plan, [])

    extracts = []
    for i, e in enumerate(ents):
        query = f"{metric} of {e}"
        here = flook == i
        dep = [p]
        if st["delegate"]:
            dep = [add(st["planner"], "decision", f"Who handles: {query}?",
                       f"delegate to {st['searcher']}", [p])]
        tool = "search"
        if st["router"]:
            bad = ftype == "wrong_tool" and here
            r_ = add("router", "decision", f"Choose a tool for: {query}",
                     "route: calculator" if bad else "route: search", dep, is_fault=bad)
            dep = [r_]
            tool = "calculator" if "calculator" in out(r_) else "search"
        elif ftype == "wrong_tool" and here:
            tool = "calculator"

        good = f"{e} {metric}: {kb_value(metric, e):,} (source: worldfacts.example)"
        if tool == "calculator":
            t = add(st["calc"], "tool_call", f"calculate: {query}", "", dep,
                    error="invalid expression", is_fault=not st["router"])
        else:
            if rng(f"retry{i}").random() < prof["retry"]:
                add(st["searcher"], "tool_call", f"search: {query}", "", dep, error="timeout")
            text, err, isf = good, None, False
            if ftype == "wrong_tool_result" and here:
                fr, isf = rng("wtr"), True
                if fr.random() < 0.7:
                    other = fr.choice([x for x in ENTITIES if x != e])
                    text = f"{other} {metric}: {kb_value(metric, other):,} (source: worldfacts.example)"
                else:
                    text = f"{e} {metric}: {int(mutate(kb_value(metric, e), fr)):,} (source: worldfacts.example)"
            elif ftype == "empty_search" and here:
                text, err, isf = "", "empty result", True
            t = add(st["searcher"], "tool_call", f"search: {query}", text, dep, error=err, is_fault=isf)

        src = out(t)
        found = nums(src)
        v = found[0] if found else rng(f"hall{i}").randint(1_000, 9_000_000)
        isf = ftype == "wrong_number" and fault.get("target") == "extract" and here
        if isf:
            v = mutate(v, rng("wn"))
        text = say(f"ex{i}", fmt(v), f"The {metric} of {e} is {fmt(v)}.")
        extracts.append(add(st["extractor"], "model_call",
                            f"Extract the {metric} of {e} from: {src}", text, [t], is_fault=isf))
        if rng(f"red{i}").random() < prof["redundant"]:
            add(st["searcher"], "tool_call", f"search: {query} latest estimate", good, [p])

    operands = [(nums(out(x)) or [0])[0] for x in extracts]
    isf = False
    if ftype == "skipped_step":
        v = mutate(compute(op, operands), rng("skip"))
        inp = f"Task: {task}\nValues: {', '.join(fmt(o) for o in operands)}"
        deps, isf = extracts, True
    else:
        dep = list(extracts)
        if st["router"]:
            dep.append(add("router", "decision", f"Choose a tool for: compute the {op}",
                           "route: calculator", [p]))
        last = add(st["calc"], "tool_call", f"calculate: {expression(op, operands)}",
                   fmt(compute(op, operands)), dep)
        if st["reviewer"]:
            got = nums(out(last))
            shown = fmt(got[0]) if got else "nothing"
            last = add("reviewer", "model_call", f"Review the result {out(last)} for: {task}",
                       f"Result {shown} looks consistent with the lookups.", [last])
        got = nums(out(last))
        v = got[0] if got else rng("hallans").randint(1_000, 9_000_000)
        inp = f"Task: {task}\nResult: {out(last)}"
        deps = [last]
    if ftype == "wrong_number" and fault.get("target") == "answer":
        v, isf = mutate(v, rng("wn")), True
    add(st["answer"], "model_call", inp,
        say("ans", f"The answer is {fmt(v)}.",
            f"Based on the lookups and the calculation, the answer is {fmt(v)}."),
        deps, is_fault=isf)

    final = nums(steps[-1]["output"])
    ok = bool(final) and abs(final[0] - truth(spec)) <= 0.011
    return dict(run_id=run_id, framework=spec["framework"], model=spec["model"], task=task,
                outcome="success" if ok else "fail",
                faulty_step=None if ok else faulty[0],
                fault_type=ftype, steps=steps, sim_spec=spec)


def descendants(run, step_no):
    """All steps that depend, directly or not, on step_no."""
    hit, changed = {step_no}, True
    while changed:
        changed = False
        for s in run["steps"]:
            if s["step_no"] not in hit and hit & set(s.get("depends_on") or []):
                hit.add(s["step_no"])
                changed = True
    return sorted(hit - {step_no})


def replay(run, step_no, new_output):
    """Stand-in for Part 1's /replay: edit one step, rerun only what depends on it."""
    new = execute(run["sim_spec"], f"{run['run_id']}_replay_{step_no}", {step_no: new_output})
    new["source_run_id"] = run.get("source_run_id")
    rerun = descendants(new, step_no)
    total = sum(s["tokens"] for s in new["steps"]) or 1
    spent = sum(s["tokens"] for s in new["steps"] if s["step_no"] in rerun)
    return dict(new_run=new,
                outcome_flipped=run["outcome"] == "fail" and new["outcome"] == "success",
                steps_rerun=len(rerun), steps_reused=len(new["steps"]) - len(rerun),
                tokens_saved_pct=round(100 * (1 - spent / total), 1))
