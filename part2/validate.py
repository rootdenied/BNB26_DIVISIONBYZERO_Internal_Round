"""Check runs against contract/schema.md and report every mismatch. Nothing is fixed.

    python -m part2.validate runs
"""
import glob
import json
import os
import sys

KINDS = ("model_call", "tool_call", "decision")
RUN_FIELDS = {"run_id": str, "framework": str, "model": str, "task": str, "outcome": str, "steps": list}
STEP_FIELDS = {"step_no": int, "actor": str, "kind": str, "input": str, "output": str,
               "depends_on": list, "tokens": int}


def _is(value, typ):
    return isinstance(value, typ) and not (typ is int and isinstance(value, bool))


def check(run):
    """Return a list of mismatches for one run (empty list = matches the contract)."""
    if not isinstance(run, dict):
        return ["run is not a JSON object"]
    bad = []
    for field, typ in RUN_FIELDS.items():
        if field not in run:
            bad.append(f"missing field `{field}`")
        elif not _is(run[field], typ):
            bad.append(f"`{field}` should be {typ.__name__}, got {type(run[field]).__name__}")
    if run.get("outcome") not in ("success", "fail"):
        bad.append(f"`outcome` should be \"success\" or \"fail\", got {run.get('outcome')!r}")
    steps = run.get("steps") if isinstance(run.get("steps"), list) else []
    if "steps" in run and not steps:
        bad.append("`steps` is empty")
    numbers = [s.get("step_no") for s in steps if isinstance(s, dict)]
    if "faulty_step" not in run:
        bad.append("missing field `faulty_step` (use null when not known)")
    elif run["faulty_step"] is not None:
        if not _is(run["faulty_step"], int):
            bad.append(f"`faulty_step` should be int or null, got {type(run['faulty_step']).__name__}")
        elif run["faulty_step"] not in numbers:
            bad.append(f"`faulty_step` {run['faulty_step']} is not a step_no in this run")
    if run.get("outcome") == "success" and run.get("faulty_step") is not None:
        bad.append("`outcome` is success but `faulty_step` is set")
    if numbers != list(range(1, len(steps) + 1)):
        bad.append(f"step_no should run 1..{len(steps)} in order, got {numbers}")
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            bad.append(f"step at position {i + 1} is not a JSON object")
            continue
        where = f"step {s.get('step_no', '?')} (position {i + 1})"
        for field, typ in STEP_FIELDS.items():
            if field not in s:
                bad.append(f"{where}: missing field `{field}`")
            elif not _is(s[field], typ):
                bad.append(f"{where}: `{field}` should be {typ.__name__}, got {type(s[field]).__name__}")
        if "error" not in s:
            bad.append(f"{where}: missing field `error` (use null when there is none)")
        elif s["error"] is not None and not isinstance(s["error"], str):
            bad.append(f"{where}: `error` should be string or null, got {type(s['error']).__name__}")
        if isinstance(s.get("kind"), str) and s["kind"] not in KINDS:
            bad.append(f"{where}: `kind` should be one of {KINDS}, got {s['kind']!r}")
        if isinstance(s.get("depends_on"), list) and _is(s.get("step_no"), int):
            for d in s["depends_on"]:
                if not _is(d, int) or d >= s["step_no"] or d < 1:
                    bad.append(f"{where}: `depends_on` entry {d!r} is not an earlier step_no")
    return bad


def check_dir(path):
    """Return {file name: [mismatches]} for every *.json in a folder, plus duplicate run_ids."""
    report, seen = {}, {}
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        name = os.path.basename(p)
        try:
            with open(p, encoding="utf-8-sig") as f:
                run = json.load(f)
        except (ValueError, UnicodeDecodeError) as e:
            report[name] = [f"not valid UTF-8 JSON: {e}"]
            continue
        report[name] = check(run)
        rid = run.get("run_id") if isinstance(run, dict) else None
        if rid in seen:
            report[name].append(f"duplicate run_id {rid!r} (also in {seen[rid]})")
        elif rid is not None:
            seen[rid] = name
    return report


def labelled_failures(path, quiet=False):
    """Real failed runs with a known faulty step that match the contract.
    Runs with mismatches are reported and left out, never patched."""
    report = check_dir(path)
    bad = {name: errs for name, errs in report.items() if errs}
    if bad and not quiet:
        print(f"contract check: {len(bad)} of {len(report)} files in {path} have mismatches and are skipped "
              f"(details: python -m part2.validate {path})")
    return [r for r in load_valid(path, report) if r.get("outcome") == "fail" and r.get("faulty_step")]


def load_valid(path, report=None):
    """Every run in the folder that matches the contract."""
    report = check_dir(path) if report is None else report
    return list(_load_ok(path, {name for name, errs in report.items() if not errs}))


def _load_ok(path, good_files):
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        if os.path.basename(p) in good_files:
            with open(p, encoding="utf-8-sig") as f:
                yield json.load(f)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs"
    report = check_dir(path)
    if not report:
        print(f"no *.json files in {path}")
        return
    bad = {name: errs for name, errs in report.items() if errs}
    for name, errs in bad.items():
        print(f"{name}: {len(errs)} mismatch(es)")
        for e in errs:
            print(f"  - {e}")
    runs = list(_load_ok(path, set(report) - set(bad)))
    failed = [r for r in runs if r["outcome"] == "fail"]
    labelled = [r for r in failed if r.get("faulty_step")]
    print(f"{len(report)} files: {len(report) - len(bad)} match the contract, {len(bad)} do not")
    print(f"usable: {len(runs) - len(failed)} success, {len(failed)} fail "
          f"({len(labelled)} with faulty_step, {len(failed) - len(labelled)} without a label)")
    no_src = [r for r in labelled if not r.get("source_run_id")]
    if no_src:
        print(f"note: {len(no_src)} labelled failed runs have no `source_run_id` (optional extra field); "
              "verify needs it to look up the good run a failure came from")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
