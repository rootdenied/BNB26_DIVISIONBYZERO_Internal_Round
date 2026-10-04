"""Start Part 1 services.

  python start.py                    # replay API :8000 + dashboard UI :8501 (ui_api.py + ui/index.html)
  python start.py --streamlit        # same, but with the old Streamlit UI (app.py) on :8501
  python start.py --mock-diagnose    # also start the mock /diagnose on :8001 (Part 2 offline)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--mock-diagnose", action="store_true")
ap.add_argument("--streamlit", action="store_true", help="use the old Streamlit UI instead of the dashboard")
a = ap.parse_args()
py = sys.executable
procs = [subprocess.Popen([py, "-m", "uvicorn", "replay_api:app", "--port", "8000"], cwd=ROOT)]
if a.mock_diagnose:
    procs.append(subprocess.Popen([py, "-m", "uvicorn", "mocks.mock_diagnose:app", "--port", "8001"], cwd=ROOT))
if a.streamlit:
    procs.append(subprocess.Popen([py, "-m", "streamlit", "run", "app.py", "--server.port", "8501",
                                   "--server.headless", "true"], cwd=ROOT))
else:
    procs.append(subprocess.Popen([py, "-m", "uvicorn", "ui_api:app", "--port", "8501"], cwd=ROOT))
print("Replay API http://localhost:8000/docs | UI http://localhost:8501"
      + (" | mock diagnose http://localhost:8001/docs" if a.mock_diagnose else ""))
try:
    while all(p.poll() is None for p in procs):
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    for p in procs:
        p.terminate()
