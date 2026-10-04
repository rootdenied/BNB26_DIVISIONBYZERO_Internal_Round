"""Start the whole Black Box demo with one command (run from the repo root).

    python run_all.py

Starts Part 2's /diagnose on :8001, then Part 1's replay API on :8000 and UI on :8501.
Open http://localhost:8501. Ctrl+C stops everything.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PART1 = ROOT / "blackbox_part 1"

if not (ROOT / "part2" / "model.txt").exists():
    sys.exit("No trained model. Run: python -m part2.make_data, then python -m part2.train")
procs = [
    subprocess.Popen([sys.executable, "-m", "uvicorn", "part2.diagnose_api:app", "--port", "8001"], cwd=ROOT),
    subprocess.Popen([sys.executable, "start.py"], cwd=PART1),
]
print("diagnose http://localhost:8001/docs | replay http://localhost:8000/docs | UI http://localhost:8501")
try:
    while all(p.poll() is None for p in procs):
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    for p in procs:
        p.terminate()
