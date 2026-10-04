"""Demo tasks + the local fact corpus the agent searches.

Why a local corpus: runs must be reproducible so fault injection and replay have a
known-good baseline. Values are approximate real-world figures (good enough for a demo).
"""
from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass, field

# attribute -> entity -> value
CORPUS: dict[str, dict[str, float]] = {
    "population": {  # millions
        "France": 68.2, "Germany": 83.3, "Italy": 58.9, "Spain": 48.1, "Japan": 124.5,
        "Canada": 40.1, "Brazil": 216.4, "Mexico": 128.5, "Egypt": 112.7, "Turkey": 85.8,
        "Poland": 36.6, "Argentina": 46.2,
    },
    "area": {  # thousand square kilometers
        "France": 551.7, "Germany": 357.6, "Italy": 301.3, "Spain": 506.0, "Japan": 377.9,
        "Canada": 9985.0, "Brazil": 8516.0, "Mexico": 1964.4, "Egypt": 1001.5, "Turkey": 783.6,
        "Poland": 312.7, "Argentina": 2780.4,
    },
    "height": {  # meters
        "Everest": 8849.0, "K2": 8611.0, "Kangchenjunga": 8586.0, "Lhotse": 8516.0,
        "Makalu": 8485.0, "Denali": 6190.0, "Kilimanjaro": 5895.0, "Elbrus": 5642.0,
        "Aconcagua": 6961.0, "Matterhorn": 4478.0,
    },
    "length": {  # kilometers
        "Nile": 6650.0, "Amazon": 6400.0, "Yangtze": 6300.0, "Mississippi": 3730.0,
        "Danube": 2850.0, "Rhine": 1230.0, "Volga": 3530.0, "Ganges": 2525.0, "Mekong": 4350.0,
    },
}

SENTENCE = {
    "population": "{e} has a population of {v} million people.",
    "area": "{e} covers a total area of {v} thousand square kilometers.",
    "height": "{e} is {v} meters tall.",
    "length": "The {e} river is {v} kilometers long.",
}
ATTR_WORDS = {
    "population": {"population", "people", "inhabitants", "populous"},
    "area": {"area", "size", "square", "large", "larger"},
    "height": {"height", "tall", "high", "taller"},
    "length": {"length", "long", "longer"},
}


def fmt_value(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".")


def fact_sentence(attr: str, entity: str) -> str:
    return SENTENCE[attr].format(e=entity, v=fmt_value(CORPUS[attr][entity]))


# ---------------------------------------------------------------- task templates
TEMPLATES = {
    "sum_pop": ("add", "population",
                "What is the combined population of {names} in millions?"),
    "diff_height": ("sub", "height", "How much taller is {a} than {b}, in meters?"),
    "ratio_area": ("div", "area",
                   "How many times larger is the area of {a} than the area of {b}? "
                   "Round to 1 decimal place."),
    "avg_length": ("avg", "length",
                   "What is the average length in kilometers of the {names} rivers?"),
}


@dataclass
class TaskSpec:
    op: str
    attr: str
    entities: list[str]


@dataclass
class Task:
    task_id: str
    text: str
    spec: TaskSpec
    expected: float = 0.0
    tolerance: float = 0.06


def _join(names: list[str]) -> str:
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]


def compute_expected(spec: TaskSpec) -> float:
    vals = [CORPUS[spec.attr][e] for e in spec.entities]
    if spec.op == "add":
        r = sum(vals)
    elif spec.op == "sub":
        r = vals[0] - vals[1]
    elif spec.op == "div":
        r = round(vals[0] / vals[1], 1)
    else:
        r = sum(vals) / len(vals)
    return round(r, 4)


def _make(task_id: str, kind: str, ents: list[str]) -> Task:
    op, attr, tpl = TEMPLATES[kind]
    text = tpl.format(names=_join(ents), a=ents[0], b=ents[-1])
    spec = TaskSpec(op, attr, ents)
    return Task(task_id, text, spec, compute_expected(spec))


def _build_tasks() -> list[Task]:
    rng = random.Random(7)
    out: list[Task] = []
    pops, areas = list(CORPUS["population"]), list(CORPUS["area"])
    peaks, rivers = list(CORPUS["height"]), list(CORPUS["length"])
    n = 0

    seen = set()

    def add(kind, ents):
        nonlocal n
        if (kind, tuple(ents)) in seen:
            return
        seen.add((kind, tuple(ents)))
        n += 1
        out.append(_make(f"t{n:02d}", kind, ents))

    for ents in [rng.sample(pops, 2) for _ in range(7)]:
        add("sum_pop", ents)
    for ents in [rng.sample(pops, 3) for _ in range(3)]:
        add("sum_pop", ents)
    for ents in [rng.sample(peaks, 2) for _ in range(5)]:
        add("diff_height", sorted(ents, key=lambda e: -CORPUS["height"][e]))
    for ents in [rng.sample(areas, 2) for _ in range(7)]:
        add("ratio_area", sorted(ents, key=lambda e: -CORPUS["area"][e]))
    for ents in [rng.sample(rivers, 3) for _ in range(4)]:
        add("avg_length", ents)
    return out


TASKS: list[Task] = _build_tasks()
# The 2 demo tasks the whole team showcases (frozen ids).
DEMO_TASK_IDS = ["t01", "t16"]


def get_task(task_id: str) -> Task | None:
    return next((t for t in TASKS if t.task_id == task_id), None)


def get_task_by_text(text: str) -> Task | None:
    text = text.strip()
    return next((t for t in TASKS if t.text == text), None)


# ---------------------------------------------------------------- parsing / checking
_PARSERS = [
    (re.compile(r"combined population of (.+?) in millions"), "add", "population"),
    (re.compile(r"How much taller is (.+?) than (.+?), in meters"), "sub", "height"),
    (re.compile(r"area of (.+?) than the area of (.+?)\?"), "div", "area"),
    (re.compile(r"average length in kilometers of the (.+?) rivers"), "avg", "length"),
]


def parse_task(text: str) -> TaskSpec | None:
    """Recover (op, attribute, entities) from task text. Used by the mock LLM only."""
    for rx, op, attr in _PARSERS:
        m = rx.search(text)
        if m:
            raw = " and ".join(m.groups()) if len(m.groups()) > 1 else m.group(1)
            ents = [e.strip() for e in re.split(r",| and ", raw) if e.strip()]
            return TaskSpec(op, attr, ents)
    return None


_NUM = re.compile(r"(?<![A-Za-z\d.])-?\d[\d,]*\.?\d*(?![A-Za-z])")


def parse_number(text: str) -> float | None:
    m = _NUM.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "").rstrip("."))
    except ValueError:
        return None


def parse_answer(answer: str) -> float | None:
    m = re.search(r"final answer\s*:\s*(.*)", answer or "", re.I | re.S)
    if m:
        return parse_number(m.group(1))
    nums = _NUM.findall(answer or "")
    if nums:
        try:
            return float(nums[-1].replace(",", "").rstrip("."))
        except ValueError:
            return None
    return None


def check_answer(task: Task, answer: str) -> bool:
    v = parse_answer(answer)
    if v is None:
        return False
    return abs(v - task.expected) <= max(task.tolerance, 0.005 * abs(task.expected))
