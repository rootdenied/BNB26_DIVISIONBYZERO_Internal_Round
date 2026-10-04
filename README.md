# Black Box: a flight recorder for AI agents

| Folder | Owner | What it is |
|---|---|---|
| `blackbox_part 1/` | Part A | Recorder, adapters (custom + LangGraph), smart replay, dashboard UI (`ui_api.py` + `ui/index.html`) |
| `part2/` | Part B | Diagnosis model (LightGBM), `/diagnose`, evaluation, verify loop |
| `contract/` | both | Frozen run format and API shapes (Part 1 keeps its own copy in `blackbox_part 1/contract/`) |

Ports: replay API 8000 (Part 1), diagnose API 8001 (Part 2), UI 8501 (Part 1).

## Setup (once)

Python 3.10 or newer.

```powershell
git clone https://github.com/rootdenied/BNB26_DIVISIONBYZERO_Internal_Round.git
cd BNB26_DIVISIONBYZERO_Internal_Round
python -m venv .venv; .venv\Scripts\Activate.ps1      # optional
pip install -r requirements.txt
```

`requirements.txt` is the two parts' lists combined (`part2/requirements.txt`,
`blackbox_part 1/requirements.txt`). The 668 recorded runs are in git as JSON
(`blackbox_part 1/runs/`); the SQLite database is not, so the first `python run_all.py` rebuilds it
from those files (about a second).

## Run the demo

```powershell
python run_all.py          # diagnose :8001 + replay :8000 + UI :8501
```

Open http://localhost:8501. It opens on Activity: every recorded run as a feed, filters on the left
(outcome, status, injected fault, model, framework) and overview charts on the right. Click a run for
its step timeline, top suspects, explanation and edit-and-replay. "Results" shows `part2/results.md`.
"Custom" runs a query you type on a model you pick (a local Ollama model or a hosted one), with an
optional injected fault, then diagnoses, replays and verifies it; see `blackbox_part 1/README.md`.
The old Streamlit UI is still there: `cd "blackbox_part 1"; python start.py --streamlit`.

Open a run. The "Source" caption under "Top suspects" must say `diagnose API` (the Diagnose API dot
bottom-left is green). If it says `built-in mock`, Part 2's API is not running and the suspects are
Part 1's heuristics.

In a run, "Why did it fail?" → "Explain this failure" gives a plain-language explanation of the top
suspect. Leave the model box empty for the built-in explanation, or type `llama3` / `groq:<real model name>`
to have an LLM word it (set `$env:GROQ_API_KEY` before `run_all.py`).

## Rebuild the numbers (from the repo root)

```powershell
python -m part2.validate "blackbox_part 1/runs"         # contract check, fixes nothing
python -m part2.make_data
python -m part2.train --real-dir "blackbox_part 1/runs"
# with run_all.py (or Part 1's replay API) running in another terminal:
python -m part2.verify --real-dir "blackbox_part 1/runs" --replay-url http://localhost:8000/replay
python -m part2.eval --real-dir "blackbox_part 1/runs" --confirmed     # writes part2/results.md
python -m part2.eval --real-dir "blackbox_part 1/runs" --confirmed --llm llama3 --llm-n 40   # needs Ollama
python -m part2.eval --real-dir "blackbox_part 1/runs" --confirmed --llm groq:<model> --llm-n 40   # needs $env:GROQ_API_KEY
```

For the live demo only, `python -m part2.train --all --real-dir "blackbox_part 1/runs"` trains on
everything. Do not quote accuracy or verify numbers from that model.

`part2/results.md` has top-1 / top-3, precision / recall / F1 with confusion matrices, accuracy on
fault types the model never trained on, the LLM-judge baseline and the confirmed-runs before/after.

## More real runs

Model ids: `llama3`, `mistral` (Ollama), or a hosted model as `groq:<model>`, `openai:<model>`,
`gemini:<model>`, `openrouter:<model>` with that provider's key set, e.g. `$env:GROQ_API_KEY="..."`.

```powershell
cd "blackbox_part 1"
python run_batch.py --models llama3,mistral --tasks 10 --repeat 2          # real LLMs via Ollama
python run_batch.py --models groq:<model> --tasks 10 --repeat 2            # hosted LLM
python run_batch.py --models llama3,mistral --tasks 10 --framework langgraph
```

Real `mistral` and `langgraph` runs are never trained on, so they become the unseen-setup test.

## Tests

```powershell
cd "blackbox_part 1"; python -m pytest -q tests      # Part 1, 41 tests, offline
```

See `part2/README.md` and `blackbox_part 1/README.md` for details, and `part2/results.md` for the numbers.
