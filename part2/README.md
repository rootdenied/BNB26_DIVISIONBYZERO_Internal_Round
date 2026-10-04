# Part 2: diagnosis model and evaluation

Run everything from the repo root.

```bash
pip install -r part2/requirements.txt
python -m part2.make_data        # 300 good runs -> 900 labelled failed runs
python -m part2.train            # trains model.txt on seen setups only
python -m part2.eval             # writes part2/results.md
python -m part2.verify           # verify loop with the built-in simulated replay
uvicorn part2.diagnose_api:app --port 8001
```

| File | What it does |
|---|---|
| sim_agent.py | Simulated agent (3 framework styles, 3 models) that writes runs in the contract format. Lets Part 2 work before Part 1 exists. |
| faults.py | Fake-failure maker. `make_failure` for simulated runs, `break_offline` for Part 1's recorded runs. |
| signals.py | Framework-agnostic signals per step, plus the evidence labels. |
| model.py | LightGBM training and `diagnose()` (top 3 with confidence and evidence). |
| diagnose_api.py | `POST /diagnose`. |
| baseline.py | Heuristic baseline and LLM-judge baseline (Ollama). |
| eval.py | Top-1 / top-3, seen vs unseen setup, per fault type. |
| verify.py | Tests top suspects through `/replay`; saves confirmed runs. |
| validate.py | Checks runs against the contract and lists every mismatch. |
| mock_replay_api.py | Stand-in for Part 1's `/replay` (simulated runs only). |

## Working with Part 1 (`blackbox_part 1/`)

Part 1 saves its runs in `blackbox_part 1/runs/`. From the repo root:

```bash
python -m part2.validate "blackbox_part 1/runs"         # lists every contract mismatch; fixes nothing
python -m part2.train --real-dir "blackbox_part 1/runs" # simulated runs + the training part of the real runs
python -m part2.verify --real-dir "blackbox_part 1/runs" --replay-url http://localhost:8000/replay
python -m part2.eval --real-dir "blackbox_part 1/runs" --confirmed
```

- Files that do not match the contract are reported and skipped, never patched.
- Real runs are grouped by task (or `source_run_id` when present) and half the tasks are test tasks.
  `train` uses seen setups on training tasks. `eval` and `verify` only use test tasks. Real
  `langgraph` and `mistral` runs are never trained on.
- `verify` copies the fix from a successful run of the same task, framework and model. With none
  available it asks Part 1's replay for a plain retry of the step (`{"rerun": true}`).
- Part 1 stores the fault type in `fault.type`; it is used for report tables only, never as a model input.
- `eval --confirmed` adds confirmed runs from half of the tasks to training and tests on the rest.
- LLM baseline: `ollama pull llama3`, then add `--llm llama3 --llm-n 40` to the eval command.
  Each test set gets an "LLM judge" row and a "same sample" row for our model.
- `python -m part2.train --all --real-dir "blackbox_part 1/runs"` trains the demo model on everything.
  Do not quote accuracy or verify numbers from it.

## Failure explanation (`POST /explain`)

`explain.py` turns the model's suspects and evidence into a plain-language explanation: summary,
root cause, how it spread, suggested fix. The trained model still picks the step; an LLM only
words the explanation and is told not to pick another step.

- Body: the run, or `{"run": {...}, "llm": "groq:<model>"}`. Response: `{"run_id", "suspects", "explanation"}`.
- No `llm` and no `BLACKBOX_EXPLAIN_LLM` set: the explanation is built from the evidence, no LLM call.
- `llm` can be an Ollama model (`llama3`) or a hosted one (`groq:<model>`, `openai:<model>`, ...).
- If the LLM cannot be reached or returns something unusable, the built-in text is returned and
  `explanation.source` says why.

## How a step gets ranked

1. **Stage 1, model.** LightGBM scores every step from 32 signals (`signals.FEATURES`). The 25 older
   ones are kept as `signals.BASELINE_FEATURES` so the previous model can be retrained and compared.
2. **Stage 2, decisive evidence.** A step whose calculation contradicts its own input, or that wrote
   word for word a request a tool then rejected, is ranked first (`signals.DECISIVE`,
   `model.final_scores`). On the training runs this evidence fired on 42 faulty steps and no healthy one.
3. **Flag.** A suspect is `flagged` when its score reaches `model.THRESHOLD` (0.8, chosen on validation).

`/diagnose` returns the same fields as before plus `flagged`. An older `model.txt` still loads.

## Data roles (never mixed)

| Role | What | Used for |
|---|---|---|
| Simulated train | seen-setup simulated runs, 4/5 of the source runs | training |
| Real train | Part 1 runs on training tasks, seen setups | training |
| Validation | grouped 5-fold cross-validation inside real train, whole tasks per fold | choosing signals, reranker, threshold |
| Replay-confirmed | `data/confirmed.jsonl` from the verify loop | `train --confirmed` only takes those from training tasks; held-out ones are refused |
| Held-out | simulated crewai / mistral runs; real runs on the other half of the tasks | scoring only |

Part 1's LangGraph runs are step-for-step copies of its custom-agent runs, so they are reported
separately as twins and not counted as independent test runs.

## What results.md reports

- Top-1 / top-3 for baseline, new signals and final on every test set, with the training setup of each row.
- A real-LLM-only row, and an unseen-model row (that model's runs removed from training).
- Validation table, and the threshold table the 0.8 was chosen from.
- Precision, recall, F1 and confusion matrices on held-out data, baseline at 0.5 against final at 0.5 and 0.8.
- Per-fault tables (simulated, real, real LLM only, and real with simulated-only training).
- Unseen failure types, signal fire rates on faulty and healthy steps, confirmed-runs before/after.
- `--llm` adds an LLM-judge row on a sample, next to the final pipeline on the same sample.

```bash
python -m part2.train --real-dir "blackbox_part 1/runs"              # served model -> model.txt
python -m part2.train --real-dir "blackbox_part 1/runs" --baseline   # old feature set -> model_baseline.txt
python -m part2.eval  --real-dir "blackbox_part 1/runs" --confirmed  # report -> results.md
```

## Known limits

- All real runs come from one agent design (plan, search, read, calculate, answer) over four task
  templates. Held-out tasks are new questions, not a new kind of agent, so 99% on them will not
  carry over unchanged to a different agent.
- Only 34 independent held-out runs use a real LLM. 100% on 34 runs is a small sample.
- Fault types the model never trained on are still weak: 45% top-1 on real held-out runs (84% top-3)
  when the fault type is removed from training.
- A simulated-only model reaches 79% top-1 on real runs with the reranker (44% before). `skipped_step`
  stays at 54% there because the simulator never produces an empty tool result without an error.
- On simulated runs the new signals changed nothing at top-1 and cost 0.4 points at top-3;
  `wrong_tool_result` with a plausible wrong number is still the weak spot there (80% top-1).
- The decisive-evidence rule is only as general as its two checks: arithmetic the step can be checked
  against, and a rejected request that appears word for word in an earlier output.
- Replay-confirmed runs added nothing at top-1 in this report (95.0% before and after). All current
  confirmed runs come from held-out tasks, so none can be added to the served model without leaking.
- A replay fix that flips the run confirms the step is on the causal path, which is usually but not always the root cause.
- The "sudden change of plan" signal from the PRD is not implemented.
