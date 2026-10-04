"""Tools the demo agent can call: a local fact search and a calculator."""
from __future__ import annotations

import ast
import operator
import random
import re

import tasks

NOT_FOUND = "No results found."
_STOP = {"the", "of", "a", "an", "is", "what", "in", "and", "for", "how", "many", "much", "to",
         "river", "mount", "mountain", "country", "total", "reported"}


class ToolError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def _lookup(query: str):
    q = _tokens(query)
    best, best_score = None, 0
    for attr, ents in tasks.CORPUS.items():
        words = tasks.ATTR_WORDS[attr]
        for ent in ents:
            if not _tokens(ent) <= q:  # entity must be named in the query
                continue
            score = 10 + len(q & words)
            if score > best_score:
                best, best_score = (attr, ent), score
    return best


def search(query: str) -> str:
    hit = _lookup(query)
    if not hit:
        raise ToolError("empty_result")
    return tasks.fact_sentence(*hit)


def distractor(query: str, seed: int = 0) -> str:
    """A plausible but WRONG result: same attribute, a different entity."""
    hit = _lookup(query)
    if not hit:
        return tasks.fact_sentence("population", "France")
    attr, ent = hit
    others = [e for e in tasks.CORPUS[attr] if e != ent]
    return tasks.fact_sentence(attr, random.Random(seed).choice(others))


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("invalid_expression")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return tasks.fmt_value(float(_eval(tree)))
    except ToolError:
        raise
    except ZeroDivisionError:
        raise ToolError("division_by_zero")
    except Exception:
        raise ToolError("invalid_expression")


TOOLS = {"search": search, "calculator": calculator}
ACTOR = {"search": "search_agent", "calculator": "calculator"}


def run_tool(name: str, arg: str) -> tuple[str, str | None]:
    """Return (output, error). Errors are recorded, never raised, so runs stay replayable."""
    fn = TOOLS.get(name)
    if fn is None:
        return f"ERROR: unknown_tool {name}", "unknown_tool"
    try:
        return fn(arg), None
    except ToolError as e:
        return (NOT_FOUND if e.code == "empty_result" else f"ERROR: {e.code}"), e.code
