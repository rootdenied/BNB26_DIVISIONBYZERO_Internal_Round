"""Verify loop: test the top suspects with replay; a suspect whose fix flips the run is confirmed.

    python -m part2.verify                                             # simulated runs, built-in simulated replay
    python -m part2.verify --real-dir runs --replay-url http://localhost:8000/replay   # Part 1's runs and replay

The fix tried on a suspect is that step's output in the good run the failure was made from
(`source_run_id`, else a successful run of the same task, framework and model). With no good run
to copy from, the fix is a plain retry of the step (`{"rerun": true}`, Part 1's replay only).
"""
import argparse
import json
import os
import random
import urllib.error
import urllib.request

from . import model, sim_agent, validate
from .data import DATA, is_unseen, read_jsonl, real_parts, write_jsonl


def reference_output(clean, step):
    """Output of the matching step in the good run: same position if it is the same
    kind of step, otherwise the nearest step with the same actor and kind."""
    at = [s for s in clean["steps"] if s["step_no"] == step["step_no"] and s["kind"] == step["kind"]]
    if at:
        return at[0]["output"]
    same = [s for s in clean["steps"] if (s["actor"], s["kind"]) == (step["actor"], step["kind"])]
    if not same:
        return None
    return min(same, key=lambda s: abs(s["step_no"] - step["step_no"]))["output"]


def call_replay(url, run, step_no, edit, n_runs=3):
    if not url:
        return sim_agent.replay(run, step_no, edit["output"])
    body = json.dumps({"run_id": run["run_id"], "step_no": step_no, "edit": edit, "n_runs": n_runs}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def setup_key(run):
    return (run.get("task"), run.get("framework"), run.get("model"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-url")
    ap.add_argument("--real-dir", help="test Part 1's real failed runs instead of simulated ones")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n-runs", type=int, default=3, help="replays per fix (real replay only; LLM answers vary)")
    a = ap.parse_args()

    booster = model.load()
    same_setup = {}
    clean = {r["run_id"]: r for r in read_jsonl(os.path.join(DATA, "clean.jsonl"))}
    if a.real_dir:
        good_real = [r for r in validate.load_valid(a.real_dir) if r["outcome"] == "success"]
        clean.update({r["run_id"]: r for r in good_real})
        same_setup = {setup_key(r): r for r in good_real}
        _, seen_part, unseen_part = real_parts(validate.labelled_failures(a.real_dir))
        pool = seen_part + unseen_part
        what = f"REAL runs from {a.real_dir}, held-out tasks only"
        if not a.replay_url:
            raise SystemExit("--real-dir needs --replay-url (Part 1's replay, e.g. http://localhost:8000/replay)")
    else:
        pool = [r for r in read_jsonl(os.path.join(DATA, "failed.jsonl")) if is_unseen(r)]
        what = "SIMULATED runs, unseen setups only"
        if a.replay_url:
            print("note: sending simulated run_ids to a real replay only works with part2.mock_replay_api; "
                  "use --real-dir for Part 1's replay")
    runs = random.Random(3).sample(pool, min(a.n, len(pool)))
    if not runs:
        print("no labelled failed runs to test")
        return

    fixed = first_try = root = replays = retries = errors = 0
    saved, rerun_steps, total_steps, confirmed = [], 0, 0, []
    for run in runs:
        good = clean.get(run.get("source_run_id")) or same_setup.get(setup_key(run))
        by_no = {s["step_no"]: s for s in run["steps"]}
        for rank, s in enumerate(model.diagnose(booster, run)):
            step = by_no[s["step_no"]]
            if good:
                fix = reference_output(good, step)
                if fix is None or fix == step["output"]:
                    continue  # nothing to change at this step
                edit = {"output": fix}
            elif a.replay_url:
                edit, retries = {"rerun": True}, retries + 1
            else:
                continue
            try:
                res = call_replay(a.replay_url, run, s["step_no"], edit, a.n_runs)
            except (urllib.error.URLError, OSError, ValueError) as e:
                errors += 1
                print(f"replay failed for {run['run_id']} step {s['step_no']}: {e}")
                continue
            replays += 1
            if res.get("outcome_flipped"):
                fixed += 1
                first_try += rank == 0
                root += s["step_no"] == run["faulty_step"]
                if res.get("tokens_saved_pct") is not None:
                    saved.append(float(res["tokens_saved_pct"]))
                rerun_steps += res.get("steps_rerun") or 0
                total_steps += (res.get("steps_rerun") or 0) + (res.get("steps_reused") or 0)
                confirmed.append({**run, "confirmed_step": s["step_no"]})
                break
    n = len(runs)
    write_jsonl(os.path.join(DATA, "confirmed.jsonl"), confirmed)
    print(f"runs tested: {n} ({what}); replay: {a.replay_url or 'built-in simulated replay'}")
    if retries:
        print(f"replays that used a plain retry because no good run was found to copy a fix from: {retries}")
    if errors:
        print(f"replay calls that failed: {errors}")
    print(f"fix success rate (a top-3 suspect's fix flips the run): {100 * fixed / n:.1f}%")
    print(f"fixed on the first suspect: {100 * first_try / n:.1f}%")
    print(f"confirmed step is the true root cause: {100 * root / max(1, fixed):.1f}% of fixed runs")
    print(f"replays per run: {replays / n:.2f}")
    print(f"tokens saved vs full rerun: {sum(saved) / max(1, len(saved)):.1f}% on average over {len(saved)} fixed runs")
    if total_steps:
        print(f"steps rerun on fixed runs: {100 * rerun_steps / total_steps:.1f}% (the rest reused saved answers)")
    print(f"{len(confirmed)} confirmed runs -> part2/data/confirmed.jsonl")


if __name__ == "__main__":
    main()
