# Contract between Part 1 and Part 2 (frozen)

Neither side changes this without telling the other.

## Run format

One JSON object per run. See `sample_runs/` for three examples.

| Field | Type | Notes |
|---|---|---|
| run_id | string | unique |
| framework | string | e.g. custom, langgraph |
| model | string | e.g. llama3 |
| task | string | the question the agent was given |
| outcome | "success" or "fail" | |
| faulty_step | int or null | the step that caused the failure, when known |
| steps | list | in order, step_no starts at 1 |

Each step: `step_no` (int), `actor` (string), `kind` (`model_call`, `tool_call` or `decision`),
`input` (string), `output` (string), `depends_on` (list of earlier step_no), `error` (string or null),
`tokens` (int).

`depends_on` rule: step N depends on every earlier step whose output it used.

Extra fields are allowed and ignored by the other side (Part 2 adds `fault_type`, `source_run_id`).

## POST /diagnose (Part 2 serves on port 8001, Part 1's UI calls it)

Request: the run object, or `{"run": {...}}`.

Response:
```json
{"run_id": "run_0142",
 "suspects": [{"step_no": 7, "confidence": 0.71, "score": 0.98,
               "evidence": ["tool_error_never_retried", "empty_result"]}]}
```
Up to 3 suspects, most likely first. `confidence` is the suspect's share of the blame within the run.

## POST /replay (Part 1 serves on port 8000, Part 2's verify loop calls it)

Request: `{"run_id": "run_0142", "step_no": 7, "edit": {"output": "new value"}}`

Response:
```json
{"new_run": {...}, "outcome_flipped": true, "steps_rerun": 6, "steps_reused": 17, "tokens_saved_pct": 74.0}
```

## Fault injection (Part 1's agent, for realistic training data)

The agent accepts `fault={"step_no": 4, "type": "empty_search"}` and runs with that step broken.
Types: wrong_tool_result, empty_search, wrong_number, skipped_step, wrong_tool.
The saved run has `outcome: "fail"` and `faulty_step` set.
