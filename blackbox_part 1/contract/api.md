# Black Box shared contract (frozen, change only with both teammates' agreement)

## 1. run.json (written by Part 1, read by Part 2)
Schema: `run.schema.json`. The required fields are exactly the agreed contract:
`run_id, framework, model, task, outcome, faulty_step, steps[step_no, actor, kind, input, output, depends_on, error, tokens]`.

* `kind` ∈ `model_call | tool_call | decision`
* `outcome` ∈ `success | fail`; `faulty_step` is an int when known (injected fault), else `null`
* `step_no` is 1..N in order; `depends_on` only points at earlier steps
* `error` examples: `empty_result`, `invalid_expression`, `unknown_tool`, `llm_error: ...`, `null`
* **Additive optional fields** (Part 2 may ignore them): `task_id`, `final_answer`, `created_at`,
  `fault` `{step_no,type,seed,applied}`, `parent_run_id` and `replay` (only on replayed runs).
  Don't feed `fault` into the model's features, because it's the label.

**depends_on rule:** step N depends on the steps the agent explicitly passed in, plus any
earlier step whose output text (≥ 12 chars; shorter words such as "unknown" occur in prompt templates) appears in N's input, or whose numbers (≥ 3 chars,
commas stripped) appear in N's input.

Handoff: Part 1 writes `runs/run_XXXX.json` (recorded runs only). Replays go to `replays/`
so they don't pollute the training set.

## 2. POST /diagnose (Part 2 builds, port 8001, Part 1's UI calls it)
Request body: a full run. Response (`diagnose.schema.json`), top 3:
```json
{"suspects": [{"step_no": 7, "confidence": 0.71, "evidence": ["tool_error", "result_ignored"]}]}
```
Evidence vocabulary used by the mock: `tool_error, empty_result, repeated_action, result_ignored,
contradiction, upstream_of_error, many_dependents, position`. The UI displays any string.

## 3. POST /replay (Part 1 builds, port 8000, Part 2's verify loop calls it)
Request:
```json
{"run_id": "run_0142", "step_no": 7, "edit": {"output": "new value"}, "n_runs": 3}
```
* `edit` takes exactly one of the following:
  * `{"output": "..."}` replaces the step's output
  * `{"input": "..."}` reruns the step with a new input
  * `{"rerun": true}` retries the step as-is. This is a generic "fix" for the verify loop.
* `run` (optional): an inline run object, for runs not in Part 1's store (e.g. Part 2's synthetic runs).
* `n_runs` (optional, default 3, max 10): each replay is repeated because LLM answers vary.

Response:
```json
{"new_run": {...}, "outcome_flipped": true, "steps_rerun": 6, "steps_reused": 17, "tokens_saved_pct": 74,
 "original_run_id": "run_0142", "n_runs": 3, "success_rate": 1.0,
 "replays": [{"run_id": "run_0142_rp001", "outcome": "success", "steps_rerun": 6, "steps_reused": 17, "tokens_saved_pct": 74.2}]}
```
* `outcome_flipped` is true when the majority outcome of the replays differs from the original's outcome.
* `steps_rerun` counts the edited step plus every re-executed step. `steps_reused` counts the saved outputs used as-is.
* `tokens_saved_pct` is 100 × (tokens of reused steps) / (tokens of all steps in the new run).
* The figures are averages over `n_runs`. `new_run` is the first successful replay, or the first replay if none succeeded.
* Errors: 404 means an unknown run_id. 422 means a bad step_no or edit, or an unknown task.

Other Part 1 endpoints: `GET /health`, `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/rewind/{step}`, `POST /record`.
