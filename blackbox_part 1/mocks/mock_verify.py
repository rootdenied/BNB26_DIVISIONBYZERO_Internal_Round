"""Mock of Part 2's verify loop: diagnose each failed run, replay the top-3 suspects
through Part 1's /replay, and report what Part 2's results table will report.

  python -m mocks.mock_verify --replay-url http://localhost:8000 --diagnose-url http://localhost:8001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-url", default="http://localhost:8000")
    ap.add_argument("--diagnose-url", default="http://localhost:8001")
    ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parent.parent / "runs"))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--n-runs", type=int, default=1)
    a = ap.parse_args(argv)

    files = sorted(Path(a.runs_dir).glob("run_*.json"))
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    failed = [r for r in runs if r["outcome"] == "fail" and r.get("faulty_step")][: a.limit]
    top1 = top3 = fixed = 0
    savings = []
    for run in failed:
        sus = requests.post(f"{a.diagnose_url}/diagnose", json=run, timeout=60).json()["suspects"]
        ranks = [s["step_no"] for s in sus]
        top1 += ranks[:1] == [run["faulty_step"]]
        top3 += run["faulty_step"] in ranks
        confirmed = None
        for s in sus:
            res = requests.post(f"{a.replay_url}/replay", json={
                "run_id": run["run_id"], "step_no": s["step_no"], "edit": {"rerun": True},
                "n_runs": a.n_runs}, timeout=300).json()
            savings.append(res["tokens_saved_pct"])
            if res["outcome_flipped"]:
                confirmed = s["step_no"]
                break
        fixed += confirmed is not None
        print(f"{run['run_id']}: truth={run['faulty_step']} suspects={ranks} confirmed={confirmed}")
    n = len(failed) or 1
    report = {"failed_runs": len(failed), "top1": round(top1 / n, 3), "top3": round(top3 / n, 3),
              "fix_success_rate": round(fixed / n, 3),
              "avg_tokens_saved_pct": round(sum(savings) / len(savings), 1) if savings else 0}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
