"""Validate run.json files / API payloads against the frozen contract.

  python contract/validate.py runs/            # every run_*.json in a folder
  python contract/validate.py contract/samples/sample_success.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
RUN_SCHEMA = json.loads((HERE / "run.schema.json").read_text(encoding="utf-8"))
DIAG_SCHEMA = json.loads((HERE / "diagnose.schema.json").read_text(encoding="utf-8"))
_REPLAY = json.loads((HERE / "replay.schema.json").read_text(encoding="utf-8"))
REPLAY_REQ = {**_REPLAY["$defs"]["request"], "$defs": _REPLAY["$defs"]}
REPLAY_RES = {**_REPLAY["$defs"]["response"], "$defs": _REPLAY["$defs"]}


def validate_run(run: dict) -> None:
    jsonschema.validate(run, RUN_SCHEMA)
    nos = [s["step_no"] for s in run["steps"]]
    if nos != list(range(1, len(nos) + 1)):
        raise ValueError(f"{run['run_id']}: step_no must be 1..N in order")
    for s in run["steps"]:
        bad = [d for d in s["depends_on"] if d >= s["step_no"]]
        if bad:
            raise ValueError(f"{run['run_id']} step {s['step_no']}: depends_on must point to earlier steps {bad}")
    if run["faulty_step"] is not None and run["faulty_step"] not in nos:
        raise ValueError(f"{run['run_id']}: faulty_step not a step")


def validate_diagnose(resp: dict) -> None:
    jsonschema.validate(resp, DIAG_SCHEMA)


def validate_replay_request(req: dict) -> None:
    jsonschema.validate(req, REPLAY_REQ)


def validate_replay_response(resp: dict) -> None:
    jsonschema.validate(resp, REPLAY_RES)
    validate_run(resp["new_run"])


if __name__ == "__main__":
    paths = []
    for a in sys.argv[1:] or [str(HERE / "samples")]:
        p = Path(a)
        paths += sorted(p.glob("*.json")) if p.is_dir() else [p]
    bad = 0
    for p in paths:
        try:
            validate_run(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"INVALID {p.name}: {e}")
    print(f"{len(paths) - bad}/{len(paths)} valid")
    sys.exit(1 if bad else 0)
