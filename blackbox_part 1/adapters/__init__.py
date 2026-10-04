"""Agent adapters. Each adapter exposes .framework and .run(task_text, recorder) -> answer."""
from __future__ import annotations


def get_agent(framework: str, llm):
    if framework == "custom":
        from adapters.custom_agent import CustomAgent
        return CustomAgent(llm)
    if framework == "langgraph":
        from adapters.langgraph_agent import LangGraphAgent
        return LangGraphAgent(llm)
    raise ValueError(f"unknown framework: {framework}")


FRAMEWORKS = ["custom", "langgraph"]
