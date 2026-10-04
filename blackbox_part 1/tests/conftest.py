import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_tmp = tempfile.mkdtemp(prefix="bb_test_")
os.environ["BLACKBOX_DB"] = os.path.join(_tmp, "test.db")
os.environ["BLACKBOX_RUNS_DIR"] = os.path.join(_tmp, "runs")
os.environ["BLACKBOX_REPLAYS_DIR"] = os.path.join(_tmp, "replays")
os.environ["BLACKBOX_CONFIRMED_DIR"] = os.path.join(_tmp, "confirmed")
os.environ["DIAGNOSE_URL"] = "http://127.0.0.1:9"   # nothing there -> UI uses mock fallback
os.environ["REPLAY_URL"] = "http://127.0.0.1:9"
