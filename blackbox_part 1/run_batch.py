"""Run the demo tasks many times on 2+ models and export run.json files for Part 2.

  python run_batch.py --models mock-large,mock-small            # offline, seconds
  python run_batch.py --models llama3,mistral --tasks 10        # real Ollama models

For every task x model: one clean run, then (with --faults) one run per fault type at a
random eligible step. Faults that don't break the run are dropped unless --keep-harmless.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

import inject
import runner
import tasks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mock-large,mock-small")
    ap.add_argument("--framework", default="custom", help="custom | langgraph")
    ap.add_argument("--tasks", type=int, default=len(tasks.TASKS), help="first N tasks")
    ap.add_argument("--repeat", type=int, default=1, help="clean runs per task (real LLMs vary)")
    ap.add_argument("--faults", type=int, default=1, help="faulty runs per fault type per clean run")
    ap.add_argument("--keep-harmless", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    stats = Counter()
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        for task in tasks.TASKS[: a.tasks]:
            for _ in range(a.repeat):
                clean = runner.record_run(task, model, a.framework)
                stats[f"clean_{clean['outcome']}"] += 1
                if clean["outcome"] != "success":
                    continue
                for _ in range(a.faults):
                    for fault in inject.plan_faults(clean, rng):
                        r = runner.record_run(task, model, a.framework, fault=fault,
                                              save=False)
                        if r["outcome"] == "fail" or a.keep_harmless:
                            import store
                            store.save_run(r)
                            stats[f"fault_{fault['type']}_{r['outcome']}"] += 1
                        else:
                            stats["fault_harmless_dropped"] += 1
            print(f"[{model}] {task.task_id} done", flush=True)
    import store
    print("\nSummary:", dict(stats))
    print(f"run.json files in: {store.RUNS_DIR}")
    return stats


if __name__ == "__main__":
    main()
