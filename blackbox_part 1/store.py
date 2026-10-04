"""Run store: SQLite (runs + steps tables) and run.json export for Part 2."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("BLACKBOX_DB", ROOT / "data" / "blackbox.db"))
RUNS_DIR = Path(os.environ.get("BLACKBOX_RUNS_DIR", ROOT / "runs"))        # handoff to Part 2
REPLAYS_DIR = Path(os.environ.get("BLACKBOX_REPLAYS_DIR", ROOT / "replays"))
CORE = ("run_id", "framework", "model", "task", "outcome", "faulty_step", "steps")


def _conn(db=None):
    p = Path(db or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS runs(
        run_id TEXT PRIMARY KEY, framework TEXT, model TEXT, task TEXT, task_id TEXT,
        outcome TEXT, faulty_step INTEGER, parent_run_id TEXT, created_at TEXT, meta TEXT);
    CREATE TABLE IF NOT EXISTS steps(
        run_id TEXT, step_no INTEGER, actor TEXT, kind TEXT, input TEXT, output TEXT,
        depends_on TEXT, error TEXT, tokens INTEGER, PRIMARY KEY(run_id, step_no));
    """)
    return c


def _next_id(c, parent=None) -> str:
    if parent:
        n = c.execute("SELECT COUNT(*) FROM runs WHERE parent_run_id=?", (parent,)).fetchone()[0]
        return f"{parent}_rp{n + 1:03d}"
    rows = c.execute("SELECT run_id FROM runs WHERE parent_run_id IS NULL").fetchall()
    nums = [int(r[0].split("_")[1]) for r in rows if r[0].count("_") == 1 and r[0].split("_")[1].isdigit()]
    return f"run_{(max(nums) + 1) if nums else 1:04d}"


def save_run(run: dict, db=None, export=True) -> str:
    """Persist a run (assigns run_id if missing) and export run.json."""
    parent = run.get("parent_run_id")
    id_mode = run.pop("_id_mode", "auto")
    with _conn(db) as c:
        for _ in range(5):
            if not run.get("run_id"):
                run["run_id"] = _next_id(c, parent)
            meta = {k: v for k, v in run.items() if k not in CORE}
            try:
                c.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    run["run_id"], run["framework"], run["model"], run["task"], run.get("task_id"),
                    run["outcome"], run.get("faulty_step"), parent, run.get("created_at"), json.dumps(meta)))
                break
            except sqlite3.IntegrityError:
                if id_mode == "auto":
                    run["run_id"] = None
                    continue
                raise
        c.executemany("INSERT OR REPLACE INTO steps VALUES(?,?,?,?,?,?,?,?,?)", [
            (run["run_id"], s["step_no"], s["actor"], s["kind"], s["input"], s["output"],
             json.dumps(s.get("depends_on", [])), s.get("error"), s.get("tokens", 0)) for s in run["steps"]])
    if export:
        export_json(run, REPLAYS_DIR if parent else RUNS_DIR)
    return run["run_id"]


def get_run(run_id: str, db=None) -> dict | None:
    with _conn(db) as c:
        r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not r:
            return None
        steps = [{
            "step_no": s["step_no"], "actor": s["actor"], "kind": s["kind"], "input": s["input"],
            "output": s["output"], "depends_on": json.loads(s["depends_on"] or "[]"),
            "error": s["error"], "tokens": s["tokens"] or 0,
        } for s in c.execute("SELECT * FROM steps WHERE run_id=? ORDER BY step_no", (run_id,))]
    run = {"run_id": r["run_id"], "framework": r["framework"], "model": r["model"], "task": r["task"],
           "outcome": r["outcome"], "faulty_step": r["faulty_step"], "steps": steps}
    run.update(json.loads(r["meta"] or "{}"))
    return run


def list_runs(outcome=None, include_replays=False, limit=500, db=None) -> list[dict]:
    q, args = "SELECT run_id, framework, model, task, task_id, outcome, faulty_step, parent_run_id, created_at FROM runs WHERE 1=1", []
    if outcome:
        q += " AND outcome=?"
        args.append(outcome)
    if not include_replays:
        q += " AND parent_run_id IS NULL"
    q += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
    args.append(limit)
    with _conn(db) as c:
        return [dict(r) for r in c.execute(q, args)]


def export_json(run: dict, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{run['run_id']}.json"
    p.write_text(json.dumps(run, indent=2), encoding="utf-8")
    return p


def import_json(path, db=None) -> str:
    """Load an external run.json (e.g. one of Part 2's synthetic runs) into the store."""
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    run["_id_mode"] = "fixed"
    rid = save_run(run, db=db, export=False)
    return rid
