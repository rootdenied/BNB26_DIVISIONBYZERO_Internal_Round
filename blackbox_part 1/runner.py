"""Record one run of a task: build agent + recorder, run, judge outcome, save."""
from __future__ import annotations

import llm as llm_mod
import store
import tasks
from adapters import get_agent
from recorder import Recorder


def record_run(task: tasks.Task, model: str, framework: str = "custom", fault: dict | None = None,
               save: bool = True, db=None) -> dict:
    agent = get_agent(framework, llm_mod.get_llm(model))
    rec = Recorder(framework, model.removeprefix("ollama:"), task.text, task.task_id, fault=fault)
    answer = agent.run(task.text, rec)
    outcome = "success" if tasks.check_answer(task, answer) else "fail"
    run = rec.finish(answer, outcome)
    if save:
        store.save_run(run, db=db)
    return run
