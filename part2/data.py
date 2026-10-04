"""Small helpers for reading and writing runs."""
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "part2", "data")
HELD_FRAMEWORK = "crewai"   # never used for training
HELD_MODEL = "mistral"      # never used for training


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_dir(path):
    """Load every run.json in a folder (this is how Part 1's real runs come in)."""
    runs = []
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(p, encoding="utf-8-sig") as f:
            runs.append(json.load(f))
    return runs


def is_unseen(run):
    return run.get("framework") == HELD_FRAMEWORK or run.get("model") == HELD_MODEL


REAL_HELD_FRAMEWORK = "langgraph"   # real runs of this framework are never trained on


def fault_of(run):
    """Fault type for reports only (never a model input): ours is `fault_type`, Part 1's is `fault.type`."""
    return str(run.get("fault_type") or (run.get("fault") or {}).get("type") or "unknown")


def group_key(run):
    """Runs with the same key came from the same good run or task and must stay on one side of a split."""
    return str(run.get("source_run_id") or run.get("task") or run.get("run_id"))


def _half(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % 2 == 0


def confirmed_for_training(conf):
    """Confirmed runs from half of the tasks; the other half stays available for testing."""
    return [r for r in conf if _half("confirmed|" + group_key(r))]


def real_unseen(run):
    return is_unseen(run) or run.get("framework") == REAL_HELD_FRAMEWORK


def real_parts(runs):
    """Split real failed runs into (train, test_seen, test_unseen).

    Runs are grouped by the good run they came from (`source_run_id`), or by task when that
    field is missing, so a task is never both trained on and tested on. Half the tasks are test
    tasks. Train = seen setups on train tasks. The held-out framework and model are only ever
    tested, and only on test tasks. The split depends only on the task, so train, eval and
    verify all agree on it."""
    def is_test(run):
        return _half(group_key(run))
    train = [r for r in runs if not is_test(r) and not real_unseen(r)]
    test_seen = [r for r in runs if is_test(r) and not real_unseen(r)]
    test_unseen = [r for r in runs if is_test(r) and real_unseen(r)]
    return train, test_seen, test_unseen


def is_real_llm(run):
    """Recorded with an actual LLM, not Part 1's rule-based mock models."""
    return "sim_spec" not in run and not str(run.get("model", "")).startswith("mock")


def trace_key(run):
    """Fingerprint of what happened in a run, ignoring run_id and framework name."""
    body = [run.get("task"), run.get("model"), run.get("faulty_step"),
            [[s.get("actor"), s.get("kind"), s.get("input"), s.get("output"), s.get("error")] for s in run["steps"]]]
    return hashlib.md5(json.dumps(body, sort_keys=True).encode()).hexdigest()


def split_twins(runs):
    """(independent, twins): a twin is step-for-step identical to an earlier run in the list
    (Part 1's LangGraph agent records exactly what its custom agent records)."""
    seen, first, twins = set(), [], []
    for r in runs:
        k = trace_key(r)
        (twins if k in seen else first).append(r)
        seen.add(k)
    return first, twins


def cv_fold(run, k=5):
    """Validation fold of a training run. Whole tasks go to one fold."""
    return int(hashlib.md5(("cv|" + group_key(run)).encode()).hexdigest(), 16) % k


def is_heldout_group(run):
    """True when the run's task belongs to the held-out half that eval and verify test on."""
    return _half(group_key(run))
