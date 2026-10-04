"""Turn each step of a run into signals that look the same for any agent.

Nothing here reads framework, model, actor names, faulty_step or fault_type,
so the same code works on a setup the model has never seen.
"""
import ast
import math
import operator
import re
import statistics

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"[A-Za-z0-9]{2,}")
# an error that says the step was asked to do something it cannot do: the request was bad
BAD_REQUEST = re.compile(r"invalid|unknown|unsupported|malformed|not[ _]allowed|bad[ _]|syntax|parse|type[ _]?error", re.I)
# an output that carries no result
NULL_OUT = re.compile(r"^\W*(no results?( found)?|not found|unknown|unavailable|none|null|n/?a|error\b.*)\W*$", re.I)
ARITH = re.compile(r"^[\d\s.,+\-*/()]+$")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}

TERM = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")

# the feature set before the ranking work; kept so the old model can be retrained and compared
BASELINE_FEATURES = [
    "has_error", "empty_output", "error_unrecovered", "is_repeat", "retry_of_error",
    "null_output", "bad_request_error", "arith_mismatch",
    "result_ignored", "out_num_unsupported", "in_num_unsupported", "io_overlap",
    "dep_has_error", "downstream_error", "prev_error",
    "n_deps", "n_dependents", "descendants_frac", "position", "is_last",
    "is_model_call", "is_tool_call", "is_decision", "rel_in_len", "rel_out_len",
]
# added for real runs; every one is computed from the recorded steps only
NEW_FEATURES = [
    "key_terms_missing",        # share of this step's own input terms (not used by its peers) absent from its output
    "rel_io_overlap",           # io_overlap compared with peers of the same kind in the run
    "rel_out_num_unsupported",  # out_num_unsupported compared with peers of the same kind
    "dictated_bad_request",     # this step's output contains, word for word, a request a later step rejected
    "child_null_frac",          # share of the steps that used this one that errored or returned nothing
    "earlier_anomaly",          # some earlier step already errored or returned nothing
    "first_anomaly",            # this step is the first in the run to error or return nothing
]
FEATURES = BASELINE_FEATURES + NEW_FEATURES

# Evidence that settles the question on its own: the step contradicts its own input, or wrote
# the exact request that failed. Steps with it are ranked first (see model.final_scores).
# Checked on training runs only: it fired on 42 faulty steps and 0 healthy ones.
DECISIVE = [
    ("calculation_does_not_match_its_input", lambda r: r["arith_mismatch"] > 0),
    ("wrote_a_request_a_tool_rejected", lambda r: r["dictated_bad_request"] > 0),
]

# (feature, label shown as evidence, test for "this signal fired")
EVIDENCE = [
    ("error_unrecovered", "tool_error_never_retried", lambda r: r["error_unrecovered"] > 0),
    ("has_error", "tool_error", lambda r: r["has_error"] > 0),
    ("arith_mismatch", "calculation_does_not_match_its_input", lambda r: r["arith_mismatch"] > 0),
    ("dictated_bad_request", "wrote_a_request_a_tool_rejected", lambda r: r["dictated_bad_request"] > 0),
    ("key_terms_missing", "tool_result_is_about_something_else",
     lambda r: r["key_terms_missing"] >= 1 and r["is_tool_call"] and not r["null_output"]),
    ("child_null_frac", "steps_that_used_it_got_nothing", lambda r: r["child_null_frac"] > 0),
    ("first_anomaly", "first_step_to_go_wrong", lambda r: r["first_anomaly"] > 0),
    ("null_output", "no_usable_result", lambda r: r["null_output"] > 0 and not r["empty_output"]),
    ("empty_output", "empty_result", lambda r: r["empty_output"] > 0),
    ("out_num_unsupported", "output_contradicts_earlier_steps", lambda r: r["out_num_unsupported"] > 0 and not r["is_tool_call"]),
    ("in_num_unsupported", "input_not_backed_by_earlier_steps", lambda r: r["in_num_unsupported"] > 0),
    ("downstream_error", "next_step_failed_because_of_it", lambda r: r["downstream_error"] > 0),
    ("is_repeat", "repeated_action", lambda r: r["is_repeat"] > 0),
    ("result_ignored", "result_ignored_by_later_steps", lambda r: r["result_ignored"] > 0),
]


def _nums(text):
    out = set()
    for m in NUM.findall(str(text or "")):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _err(step):
    e = step.get("error")
    return bool(e) and str(e).strip().lower() not in ("none", "null", "")


def _bad_request(step):
    return _err(step) and bool(BAD_REQUEST.search(str(step.get("error"))))


def _arith_value(node):
    if isinstance(node, ast.Expression):
        return _arith_value(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_arith_value(node.left), _arith_value(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_arith_value(node.operand))
    raise ValueError


def _arith_mismatch(inp, out):
    """1 when the input is plain arithmetic (optionally after a "word:" prefix) and the
    output's number is not its value."""
    expr = inp.split(":", 1)[-1].strip().replace(",", "")
    if not expr or not ARITH.match(expr) or not re.search(r"[+\-*/]", expr.lstrip("-")):
        return 0.0
    try:
        want = float(_arith_value(ast.parse(expr, mode="eval")))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError, MemoryError):
        return 0.0
    got = NUM.findall(out)
    if not got:
        return 0.0
    try:
        have = float(got[0].replace(",", ""))
    except ValueError:
        return 0.0
    return float(abs(have - want) > 0.011 + 1e-6 * abs(want))


def _terms(text):
    return {w.lower() for w in TERM.findall(str(text or ""))}


def _null(text):
    text = str(text or "").strip()
    return not text or bool(NULL_OUT.match(text))


def _key(step):
    return (step.get("kind"), str(step.get("input") or "").strip().lower())


def extract(run):
    """Return one feature dict per step, in step order."""
    steps = run["steps"]
    n = len(steps)
    by = {s["step_no"]: s for s in steps}
    children = {k: [] for k in by}
    for s in steps:
        for d in s.get("depends_on") or []:
            if d in children:
                children[d].append(s["step_no"])

    def n_desc(k):
        seen, stack = set(), list(children[k])
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                stack.extend(children[c])
        return len(seen)

    task_nums = _nums(run.get("task"))
    rows = []
    for i, s in enumerate(steps):
        inp, out = str(s.get("input") or ""), str(s.get("output") or "")
        deps = [by[d] for d in (s.get("depends_on") or []) if d in by]
        kids = [by[c] for c in children[s["step_no"]]]
        earlier = [t for t in steps[:i] if _key(t) == _key(s)]
        later = [t for t in steps[i + 1:] if _key(t) == _key(s)]
        dep_nums = set().union(*[_nums(d.get("output")) for d in deps]) if deps else set()
        in_nums, out_nums = _nums(inp), _nums(out)
        out_bad = out_nums - in_nums - dep_nums
        in_bad = in_nums - dep_nums - task_nums
        in_words = {w.lower() for w in WORD.findall(inp)}
        out_words = {w.lower() for w in WORD.findall(out)}
        kind = s.get("kind")
        rows.append({
            "has_error": float(_err(s)),
            "empty_output": float(not out.strip()),
            "error_unrecovered": float(_err(s) and not later),
            "is_repeat": float(bool(earlier)),
            "retry_of_error": float(any(_err(t) for t in earlier)),
            "null_output": float(not out.strip() or bool(NULL_OUT.match(out.strip()))),
            "bad_request_error": float(_bad_request(s)),
            "arith_mismatch": _arith_mismatch(inp, out),
            "result_ignored": float(not kids and i < n - 1),
            "out_num_unsupported": len(out_bad) / len(out_nums) if out_nums else 0.0,
            "in_num_unsupported": len(in_bad) / len(in_nums) if in_nums else 0.0,
            "io_overlap": len(in_words & out_words) / len(in_words) if in_words else 1.0,
            "dep_has_error": float(any(_err(d) or not str(d.get("output") or "").strip() for d in deps)),
            "downstream_error": float(any(_err(k) for k in kids)),
            "prev_error": float(i > 0 and _err(steps[i - 1])),
            "n_deps": float(len(deps)),
            "n_dependents": float(len(kids)),
            "descendants_frac": n_desc(s["step_no"]) / n,
            "position": (i + 1) / n,
            "is_last": float(i == n - 1),
            "is_model_call": float(kind == "model_call"),
            "is_tool_call": float(kind == "tool_call"),
            "is_decision": float(kind == "decision"),
            "log_in_len": math.log1p(len(inp)),
            "log_out_len": math.log1p(len(out)),
        })
    # Raw text length mostly encodes how chatty a framework or model is, so the model
    # only sees length relative to the other steps of the same kind in the same run.
    for i, row in enumerate(rows):
        peers = [rows[j] for j, t in enumerate(steps) if j != i and t.get("kind") == steps[i].get("kind")]
        for name in ("in_len", "out_len"):
            med = statistics.median(p["log_" + name] for p in peers) if peers else row["log_" + name]
            row["rel_" + name] = row["log_" + name] - med
        for name in ("io_overlap", "out_num_unsupported"):
            row["rel_" + name] = row[name] - statistics.median(p[name] for p in peers) if peers else 0.0
        s = steps[i]
        inp, out = str(s.get("input") or ""), str(s.get("output") or "")
        # terms only this step was asked about: "Germany" in "Germany total population" when its
        # peers ask about other countries. A result that mentions none of them is off target.
        shared = set()
        for j, t in enumerate(steps):
            if j != i and t.get("kind") == s.get("kind"):
                shared |= _terms(t.get("input"))
        own = _terms(inp) - shared
        row["key_terms_missing"] = len(own - _terms(out)) / len(own) if own and peers else 0.0
        kids = [by[c] for c in children[s["step_no"]]]
        row["dictated_bad_request"] = float(any(
            _bad_request(k) and str(k.get("input") or "").strip() and str(k.get("input")).strip() in out
            for k in kids))
        row["child_null_frac"] = sum(1 for k in kids if _err(k) or _null(k.get("output"))) / len(kids) if kids else 0.0
        row["earlier_anomaly"] = float(any(r["has_error"] or r["null_output"] for r in rows[:i]))
        row["first_anomaly"] = float(bool(row["has_error"] or row["null_output"]) and not row["earlier_anomaly"])
    return rows


def decisive(row):
    """Labels of the decisive evidence that fired on this step (usually none)."""
    return [label for label, test in DECISIVE if test(row)]
