"""Regenerate contract/samples (2 failed + 1 successful run) from the real agent."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BLACKBOX_DB", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("BLACKBOX_RUNS_DIR", tempfile.mkdtemp())

import runner  # noqa: E402
import tasks  # noqa: E402

out = ROOT / "contract" / "samples"
out.mkdir(parents=True, exist_ok=True)
specs = [
    ("sample_success.json", "t01", None),
    ("sample_fail_wrong_number.json", "t01", {"step_no": 5, "type": "wrong_number", "seed": 1}),
    ("sample_fail_wrong_tool.json", "t16", {"step_no": 1, "type": "wrong_tool", "seed": 2}),
]
for i, (name, tid, fault) in enumerate(specs, 1):
    run = runner.record_run(tasks.get_task(tid), "mock-large", fault=fault, save=False)
    run["run_id"] = f"sample_{i:02d}"
    run["created_at"] = "2026-10-03T00:00:00+00:00"
    (out / name).write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(name, run["outcome"], "faulty_step=", run["faulty_step"])
