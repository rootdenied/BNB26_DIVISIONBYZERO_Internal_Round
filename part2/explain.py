"""Failure explanation in plain language.

The trained model decides WHICH step is the suspect. This module only explains it:
  * explain(run, suspects)            - built from the model's evidence, no LLM needed
  * explain(run, suspects, llm="...") - an LLM rewrites it with the trace in front of it;
                                        it is told the suspect and may not pick another step.
If the LLM is unreachable or returns something unusable, the built-in explanation is returned.

    llm: an Ollama model (llama3) or a hosted one (groq:<model>, openai:<model>, gemini:<model>,
    openrouter:<model>). Default: the BLACKBOX_EXPLAIN_LLM environment variable, else none.
"""
import json
import os

from . import baseline

# evidence label -> (what it means for this step, what to try)
MEANING = {
    "tool_error_never_retried": ("the tool call failed with an error and was never tried again",
                                 "retry the call, or change its input so the tool can answer"),
    "tool_error": ("the step ended with an error", "retry the step or correct its input"),
    "empty_result": ("the step returned nothing", "rerun it, or reword the request so it returns a result"),
    "no_usable_result": ("the step returned a placeholder instead of a real result",
                         "rerun it, or reword the request so it returns a result"),
    "wrote_a_request_a_tool_rejected": ("it wrote a request, word for word, that a tool then rejected as invalid",
                                        "change the request it wrote, or send it to a tool that can handle it"),
    "tool_result_is_about_something_else": ("the tool's answer does not mention the thing it was asked about",
                                            "rerun the lookup, or replace the result with one about the right subject"),
    "steps_that_used_it_got_nothing": ("the steps that relied on it ended up with no usable result",
                                       "change this step's output so the steps after it have something to work with"),
    "first_step_to_go_wrong": ("it is the first step in the run to error or return nothing",
                               "fix this step first; later problems may only be consequences"),
    "calculation_does_not_match_its_input": ("the number it returned is not what its own input works out to",
                                             "recompute the calculation"),
    "output_contradicts_earlier_steps": ("its output contains a number that none of the steps it relied on produced",
                                         "replace the output with the value found in the earlier step"),
    "input_not_backed_by_earlier_steps": ("it was given a value that no earlier step produced",
                                          "check where that value came from and pass the right one"),
    "tool_result_does_not_mention_what_was_asked": ("the tool answered about something other than what was asked",
                                                    "rerun the lookup, or replace the result with one about the right subject"),
    "next_step_failed_because_of_it": ("the step that used its output failed straight after",
                                       "change this step's output so the next step gets something it can use"),
    "repeated_action": ("it repeats an action already taken earlier in the run", "drop the repeat or change its input"),
    "result_ignored_by_later_steps": ("no later step used its result", "check whether the step was needed, or wire its result in"),
    "position_and_dependents_only": ("no single signal fired; it is ranked by its position and how many later steps depend on it",
                                     "inspect its output by hand before editing"),
}


def _short(text, n=160):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def _tail(text, n=110):
    """End of a long text: for a prompt, the template is at the start and the actual question at the end."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else "…" + text[-n:]


def _descendants(run, step_no):
    hit, changed = {step_no}, True
    while changed:
        changed = False
        for s in run["steps"]:
            if s["step_no"] not in hit and hit & set(s.get("depends_on") or []):
                hit.add(s["step_no"])
                changed = True
    return sorted(hit - {step_no})


def built_in(run, suspects):
    """Explanation assembled from the model's suspects and evidence. No LLM involved."""
    by_no = {s["step_no"]: s for s in run["steps"]}
    top = suspects[0]
    step = by_no[top["step_no"]]
    reasons = [MEANING.get(e, (e.replace("_", " "), "inspect the step")) for e in top.get("evidence", [])]
    after = _descendants(run, top["step_no"])
    last = run["steps"][-1]
    what = "; ".join(r[0] for r in reasons) or "the model ranked it first"
    root = (f"Step {step['step_no']} ({step.get('actor')}, {step.get('kind')}) is the most likely cause "
            f"({top.get('confidence', 0):.0%} of the blame): {what}. "
            f"It was asked \"{_tail(step.get('input'))}\" and returned \"{_short(step.get('output'), 110) or '(nothing)'}\""
            + (f" with error `{step['error']}`." if step.get("error") else "."))
    if after:
        spread = (f"{len(after)} later step(s) built on it (steps {', '.join(map(str, after))}), ending in the final "
                  f"output \"{_short(last.get('output'), 110)}\".")
    else:
        spread = "No later step depends on it, so it is the last link before the final output."
    fix = (f"Edit step {step['step_no']}: {reasons[0][1] if reasons else 'inspect the step'}. Then replay; only "
           + (f"steps {', '.join(map(str, after))} need to run again." if after else "that step needs to run again."))
    others = [f"step {s['step_no']} ({s.get('confidence', 0):.0%})" for s in suspects[1:]]
    return {
        "summary": f"The run failed on \"{_short(run.get('task'), 120)}\". The most likely cause is step {step['step_no']}.",
        "root_cause": root,
        "how_it_spread": spread,
        "suggested_fix": fix,
        "other_suspects": ("If that fix does not flip the run, try " + ", then ".join(others) + ".") if others else "",
        "suspect_step": step["step_no"],
        "source": "built-in (from the diagnosis model's evidence)",
    }


def _prompt(run, suspects, base):
    listed = "\n".join(f"- step {s['step_no']}: confidence {s.get('confidence')}, signals {s.get('evidence')}"
                       for s in suspects)
    return (baseline.render(run, width=400) + "\n\nA trained diagnosis model ranked these suspect steps "
            f"(most likely first):\n{listed}\n\nIts draft explanation:\n{json.dumps(base, indent=1)}\n\n"
            f"You are explaining this failure to the developer who owns the agent. Treat step {suspects[0]['step_no']} "
            "as the cause; do not pick a different step. Use only what is in the trace; if something is not in the "
            "trace, do not state it. Be specific: quote the values involved. Plain language, no jargon, at most 3 "
            "sentences per field. Reply with JSON only, with exactly these string fields: "
            '{"summary": ..., "root_cause": ..., "how_it_spread": ..., "suggested_fix": ...}')


def explain(run, suspects, llm=None):
    """Return the explanation dict. Falls back to the built-in text if the LLM cannot be used."""
    if not suspects:
        return {"summary": "No suspects: the run has no steps to blame.", "root_cause": "", "how_it_spread": "",
                "suggested_fix": "", "other_suspects": "", "suspect_step": None, "source": "built-in"}
    base = built_in(run, suspects)
    llm = (llm or os.environ.get("BLACKBOX_EXPLAIN_LLM") or "").strip()
    if not llm:
        return base
    try:
        text = baseline._ask(llm, _prompt(run, suspects, base), "http://localhost:11434/api/generate")
        got = json.loads(text[text.find("{"): text.rfind("}") + 1])
        fields = {k: str(got[k]).strip() for k in ("summary", "root_cause", "how_it_spread", "suggested_fix")}
        if not all(fields.values()):
            raise ValueError("empty field")
    except (Exception, SystemExit) as e:  # noqa: BLE001  any LLM problem: keep the built-in text
        reason = str(e) or type(e).__name__
        return {**base, "source": base["source"] + f"; {llm} was not used: {_short(reason, 120)}"}
    return {**base, **fields, "source": f"{llm}, grounded in the diagnosis model's suspects"}
