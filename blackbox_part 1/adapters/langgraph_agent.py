"""Adapter: the same research agent built as a LangGraph StateGraph.

Plugging Black Box into LangGraph = call rec.step(...) inside each node (~20 lines).
Replay works unchanged because the graph is simply re-executed with a replay Recorder.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import tasks
from adapters import prompts
from adapters.custom_agent import _llm_fn, _tool_fn, build_expression, parse_plan
import tools


class S(TypedDict, total=False):
    task: str
    plan_step: int
    plan: Optional[dict]
    idx: int
    numbers: list
    read_steps: list
    result: str
    calc_dep: list
    answer: str


class LangGraphAgent:
    framework = "langgraph"

    def __init__(self, llm):
        self.llm = llm
        self.rec: Any = None
        g = StateGraph(S)
        g.add_node("plan", self._plan)
        g.add_node("act", self._act)
        g.add_node("calc", self._calc)
        g.add_node("answer", self._answer)
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", lambda s: "act" if s.get("plan") else "answer")
        g.add_conditional_edges("act", lambda s: "act" if s["idx"] < len(s["plan"]["steps"]) else "calc")
        g.add_edge("calc", "answer")
        g.add_edge("answer", END)
        self.graph = g.compile()

    # nodes ------------------------------------------------------------
    def _plan(self, s: S) -> S:
        st = self.rec.step("planner", "decision", prompts.PLAN.format(task=s["task"]), _llm_fn(self.llm, True))
        return {"plan_step": st.step_no, "plan": parse_plan(st.output), "idx": 0, "numbers": [],
                "read_steps": [], "result": "unavailable", "calc_dep": []}

    def _act(self, s: S) -> S:
        q = s["plan"]["steps"][s["idx"]]
        tool = str(q.get("tool", "search"))
        t = self.rec.step(tools.ACTOR.get(tool, tool), "tool_call", str(q["query"]), _tool_fn(tool),
                          deps=[s["plan_step"]])
        r = self.rec.step("reader", "model_call", prompts.READ.format(question=q["query"], text=t.output),
                          _llm_fn(self.llm), deps=[t.step_no])
        return {"idx": s["idx"] + 1, "numbers": s["numbers"] + [tasks.parse_number(r.output)],
                "read_steps": s["read_steps"] + [r.step_no]}

    def _calc(self, s: S) -> S:
        out = {"calc_dep": s["read_steps"]}
        if all(n is not None for n in s["numbers"]):
            expr = build_expression(s["plan"]["expression"], s["numbers"])
            if expr:
                c = self.rec.step("calculator", "tool_call", expr, _tool_fn("calculator"),
                                  deps=[s["plan_step"], *s["read_steps"]])
                out = {"calc_dep": [c.step_no],
                       "result": c.output if not c.error and c.output else "unavailable"}
        return out

    def _answer(self, s: S) -> S:
        if not s.get("plan"):
            return {"answer": "Final answer: unknown (no valid plan)"}
        a = self.rec.step("answerer", "model_call",
                          prompts.ANSWER.format(task=s["task"], result=s.get("result", "unavailable")),
                          _llm_fn(self.llm), deps=s.get("calc_dep", []))
        return {"answer": a.output}

    def run(self, task: str, rec) -> str:
        self.rec = rec
        return self.graph.invoke({"task": task}, {"recursion_limit": 50})["answer"]
