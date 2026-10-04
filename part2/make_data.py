"""Generate training data: good runs, then labelled failed runs made from them.

    python -m part2.make_data --clean 300 --per 3
"""
import argparse
import json
import os
import random

from . import faults, sim_agent
from .data import DATA, ROOT, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=int, default=300, help="number of good runs")
    ap.add_argument("--per", type=int, default=3, help="failed runs made from each good run")
    a = ap.parse_args()

    r = random.Random(42)
    clean, failed = [], []
    for seed in range(a.clean):
        fw = sim_agent.FRAMEWORKS[seed % 3]
        model = sim_agent.MODEL_NAMES[(seed // 3) % 3]
        run = sim_agent.execute(sim_agent.make_spec(seed, fw, model), f"run_{seed:04d}")
        clean.append(run)
        for _ in range(a.per):
            bad = faults.make_failure(run, r)
            if bad:
                failed.append(bad)
    write_jsonl(os.path.join(DATA, "clean.jsonl"), clean)
    write_jsonl(os.path.join(DATA, "failed.jsonl"), failed)

    samples = os.path.join(ROOT, "contract", "sample_runs")
    os.makedirs(samples, exist_ok=True)
    for name, run in [("sample_success", clean[0]), ("sample_fail_1", failed[0]), ("sample_fail_2", failed[4])]:
        public = {k: v for k, v in run.items() if k != "sim_spec"}
        path = os.path.join(samples, name + ".json")
        if not os.path.exists(path):  # contract/ is frozen: only fill in missing samples
            with open(path, "w", encoding="utf-8") as f:
                json.dump(public, f, indent=2)
    print(f"{len(clean)} good runs, {len(failed)} labelled failed runs -> {DATA}")


if __name__ == "__main__":
    main()
