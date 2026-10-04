"""Train the model that /diagnose serves.

    python -m part2.train                 # seen simulated setups only (keeps the unseen-setup numbers honest)
    python -m part2.train --real-dir runs # also learn from the training part of Part 1's labelled failed runs
    python -m part2.train --real-dir runs --confirmed   # plus replay-confirmed runs from training tasks
    python -m part2.train --real-dir runs --baseline    # old feature set -> part2/model_baseline.txt (for comparison)
    python -m part2.train --all           # demo model: every setup (and every real run with --real-dir)

Training data is kept in four separate groups and printed: simulated, real, replay-confirmed,
and (never used here) the held-out tasks that eval and verify test on.
"""
import argparse
import os

from . import model, validate
from .data import DATA, is_heldout_group, is_unseen, read_jsonl, real_parts
from .signals import BASELINE_FEATURES, FEATURES


def confirmed_for_model(known_ids):
    """Replay-confirmed runs that are safe to train on: their task must be a training task
    (never a held-out one) and the run must not already be in training. Label = confirmed step."""
    path = os.path.join(DATA, "confirmed.jsonl")
    conf = read_jsonl(path) if os.path.exists(path) else []
    safe = [{**r, "faulty_step": r["confirmed_step"]} for r in conf
            if "sim_spec" not in r and not is_heldout_group(r) and r["run_id"] not in known_ids]
    refused = sum(1 for r in conf if "sim_spec" in r or is_heldout_group(r))
    return safe, refused, len(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--real-dir")
    ap.add_argument("--confirmed", action="store_true", help="add replay-confirmed runs from training tasks")
    ap.add_argument("--baseline", action="store_true", help="train the old feature set to model_baseline.txt")
    ap.add_argument("--real-weight", type=float, default=1.0, help="weight of real rows relative to simulated ones")
    a = ap.parse_args()
    runs = read_jsonl(os.path.join(DATA, "failed.jsonl"))
    if not a.all:
        runs = [r for r in runs if not is_unseen(r)]
    print(f"simulated: {len(runs)} failed runs")
    if a.real_dir:
        real = validate.labelled_failures(a.real_dir)
        used = real if a.all else real_parts(real)[0]
        print(f"real: {len(used)} of {len(real)} labelled failed runs from {a.real_dir}"
              + ("" if a.all else " (seen setups, training tasks; held-out tasks are never trained on)"))
        runs += used
    if a.confirmed:
        safe, refused, total = confirmed_for_model({r["run_id"] for r in runs})
        print(f"replay-confirmed: {len(safe)} of {total} added; {refused} refused because their task is held out "
              "for testing (or they are simulated); the rest were already in training")
        runs += safe
    path = model.MODEL_PATH.replace("model.txt", "model_baseline.txt") if a.baseline else model.MODEL_PATH
    feats = BASELINE_FEATURES if a.baseline else FEATURES
    model.save(model.train(runs, features=feats, real_weight=a.real_weight), path)
    print(f"trained on {len(runs)} failed runs, {len(feats)} features -> {path}")
    if a.all:
        print("note: this model has seen every setup; do not quote verify numbers from it")


if __name__ == "__main__":
    main()
