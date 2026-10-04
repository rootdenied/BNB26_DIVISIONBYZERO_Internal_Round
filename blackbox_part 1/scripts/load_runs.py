"""Rebuild the local run database from the run.json files in runs/ (and replays/ if present).

The SQLite database (data/blackbox.db) is not kept in git, so a fresh clone has the recorded
runs as JSON only. run_all.py calls this once when the database does not exist yet.

  python scripts/load_runs.py        # from "blackbox_part 1/"; does nothing if the database exists
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import store  # noqa: E402


def load() -> int:
    files = sorted(store.RUNS_DIR.glob("run_*.json")) + sorted(store.REPLAYS_DIR.glob("run_*.json"))
    n = 0
    for f in files:          # originals first, so replays find their parent run
        try:
            store.import_json(f)
            n += 1
        except Exception as e:  # noqa: BLE001  one unreadable file must not stop the demo
            print(f"skipped {f.name}: {type(e).__name__}: {e}")
    return n


if __name__ == "__main__":
    if store.DB_PATH.exists():
        print(f"{store.DB_PATH} already exists, nothing to do.")
    else:
        print(f"Loaded {load()} recorded runs into {store.DB_PATH}")
