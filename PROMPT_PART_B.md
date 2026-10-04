# Prompt: build / extend Part B of Black Box

Paste this into your coding assistant from the repo root.

---

You are helping me build Part B of "Black Box", a flight recorder for AI agents, for a hackathon
with 8 hours total. My teammate builds Part A (agent, recorder, replay, Streamlit UI). I own Part B:
the diagnosis model and the evaluation. Work only inside `part2/` and never change `contract/`.

## What Part B does

Input: a failed agent run in the shared format (`contract/schema.md`).
Output: the 3 steps most likely to have caused the failure, each with a confidence and evidence,
served at `POST /diagnose` on port 8001, plus an accuracy report.

This is a trained classifier, not an LLM. "An LLM reads the log and guesses" is the baseline we beat.

## Contract (frozen)

Run: `run_id, framework, model, task, outcome ("success"|"fail"), faulty_step, steps[]`.
Step: `step_no, actor, kind ("model_call"|"tool_call"|"decision"), input, output, depends_on[], error, tokens`.

`POST /diagnose` takes a run and returns
`{"run_id": ..., "suspects": [{"step_no", "confidence", "score", "evidence": [...]}]}` (top 3).

`POST /replay` (Part A, port 8000) takes `{"run_id", "step_no", "edit": {"output"}}` and returns
`{"new_run", "outcome_flipped", "steps_rerun", "steps_reused", "tokens_saved_pct"}`.

## Current state

A working version exists; read `part2/README.md` first. Pipeline:

1. `sim_agent.py` simulates an agent in 3 framework styles and 3 models and emits contract-format runs.
2. `faults.py` breaks one step of a good run (wrong_tool_result, empty_search, wrong_number,
   skipped_step, wrong_tool). The broken step is the label.
3. `signals.py` turns each step into framework-agnostic features. It must never read
   `framework`, `model`, actor names, `faulty_step` or `fault_type`.
4. `model.py` trains LightGBM (one row per step, label = is the faulty step) and ranks the top 3,
   using SHAP contributions to pick the evidence.
5. `eval.py` holds out framework `crewai` and model `mistral` entirely and reports top-1 / top-3
   on seen and unseen setups against a heuristic baseline and an optional LLM judge (Ollama).
6. `verify.py` sends top suspects to `/replay`; a suspect whose fix flips the outcome is confirmed.

Commands: `python -m part2.make_data`, `python -m part2.train`, `python -m part2.eval`,
`python -m part2.verify`, `uvicorn part2.diagnose_api:app --port 8001`.

## What I need from you now, in this order

1. Run the five commands above and confirm they all work on my machine. Fix anything that breaks.
2. Run the LLM-judge baseline with Ollama (`python -m part2.eval --llm llama3 --llm-n 40`) and make
   sure `results.md` shows our model and the LLM on the same sample.
3. When my teammate's real runs appear in `runs/*.json`: validate they match the contract (report
   every mismatch, do not silently fix), then retrain with `--real-dir runs` and add their numbers
   to the report. If accuracy on real runs is poor, look at which signals fail to fire on real
   steps and adapt `signals.py`; keep it framework-agnostic.
4. Point `verify.py` at the real replay (`--replay-url http://localhost:8000/replay`) and report
   fix success rate and tokens saved.
5. Only if time is left: add the confirmed runs from `part2/data/confirmed.jsonl` to training and
   show accuracy before and after.

## Rules

- Keep it small: plain Python, LightGBM, FastAPI. No new services, no notebooks, no deep learning.
- Never evaluate on runs the model trained on. Split by `source_run_id`, and keep the held-out
  framework and model out of training.
- Report numbers honestly, including where the model is weak (per fault type). Say clearly which
  numbers come from simulated runs and which from real runs.
- After every change, rerun `python -m part2.eval` and show me the table.
- Stop adding features at hour 7; after that only fix bugs.
