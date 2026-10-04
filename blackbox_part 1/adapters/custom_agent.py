"""Adapter: plain custom Python agent (plan -> search+read per fact -> calculate -> answer).

Plan-and-execute keeps independent branches independent, so editing one search only
reruns its own reader, the calculator and the answer - the other branches are reused.
"""
from __future__ import annotations

import llm as llm_mod
import tasks
import tools
from adapters import prompts


def _llm_fn(llm, json_mode=False):
    def fn(p):
        r = llm.complete(p, json_mode=json_mode)
        return r.text, r.tokens, r.error
    return fn


def _tool_fn(name):
    def fn(arg):
        out, err = tools.run_tool(name, arg)
        return out, 0, err
    return fn


def parse_plan(text: str):
    plan = llm_mod.extract_json(text)
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or not plan["steps"]:
        return None
    steps = [s for s in plan["steps"] if isinstance(s, dict) and s.get("query")][:6]
    return {"steps": steps, "expression": str(plan.get("expression", ""))} if steps else None


def build_expression(template: str, numbers: list[float]) -> str | None:
    expr = template
    for i, n in enumerate(numbers):
        expr = expr.replace(f"{{{i}}}", tasks.fmt_value(n) if n >= 0 else f"({tasks.fmt_value(n)})")
    return None if "{" in expr or not expr.strip() else expr


class CustomAgent:
    framework = "custom"

    def __init__(self, llm):
        self.llm = llm

    def run(self, task: str, rec) -> str:
        plan_s = rec.step("planner", "decision", prompts.PLAN.format(task=task), _llm_fn(self.llm, True))
        plan = parse_plan(plan_s.output)
        if plan is None:
            return "Final answer: unknown (no valid plan)"
        numbers, read_steps = [], []
        for st in plan["steps"]:
            tool = str(st.get("tool", "search"))
            t = rec.step(tools.ACTOR.get(tool, tool), "tool_call", str(st["query"]), _tool_fn(tool),
                         deps=[plan_s.step_no])
            r = rec.step("reader", "model_call",
                         prompts.READ.format(question=st["query"], text=t.output), _llm_fn(self.llm),
                         deps=[t.step_no])
            numbers.append(tasks.parse_number(r.output))
            read_steps.append(r.step_no)
        result, calc_dep = "unavailable", read_steps
        if all(n is not None for n in numbers):
            expr = build_expression(plan["expression"], numbers)
            if expr:
                c = rec.step("calculator", "tool_call", expr, _tool_fn("calculator"),
                             deps=[plan_s.step_no, *read_steps])
                result, calc_dep = (c.output if not c.error and c.output else "unavailable"), [c.step_no]
        a = rec.step("answerer", "model_call", prompts.ANSWER.format(task=task, result=result),
                     _llm_fn(self.llm), deps=calc_dep)
        return a.output
