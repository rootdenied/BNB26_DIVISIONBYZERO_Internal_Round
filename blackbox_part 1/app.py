"""Black Box UI (Part 1).  streamlit run app.py --server.port 8501"""
from __future__ import annotations

import difflib
import html
import json
import os
import re
from pathlib import Path

import streamlit as st

import clients
import inject
import replay
import runner
import store
import tasks

st.set_page_config(page_title="Black Box", page_icon="⬛", layout="wide")
CONFIRMED_DIR = Path(os.environ.get("BLACKBOX_CONFIRMED_DIR", store.ROOT / "confirmed"))
MODELS = ["mock-large", "mock-small", "llama3", "mistral"]
MODEL_HELP = ("mock-* run offline. llama3 / mistral need Ollama running locally. For a hosted model type its id "
              "below, e.g. groq:<model>, openai:<model>, gemini:<model>, openrouter:<model> "
              "(needs GROQ_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY set before starting).")

st.markdown("""
<style>
.bb-card{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:8px 12px;margin-bottom:6px}
.bb-sus{border:2px solid #e8590c;background:rgba(232,89,12,.08)}
.bb-sel{outline:3px solid #1c7ed6}
.bb-chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;margin:0 4px 2px 0;
         background:rgba(128,128,128,.18)}
.bb-err{background:rgba(224,49,49,.2)}
.bb-reused{background:rgba(47,158,68,.22)} .bb-rerun{background:rgba(240,140,0,.25)}
.bb-edited{background:rgba(28,126,214,.25)} .bb-dim{opacity:.55}
.bb-mono{font-family:ui-monospace,monospace;font-size:.82rem;white-space:pre-wrap;word-break:break-word}
.bb-del{background:rgba(224,49,49,.28);text-decoration:line-through} .bb-ins{background:rgba(47,158,68,.32)}
.bb-same{opacity:.6} .bb-changed{border-left:4px solid #f08c00} .bb-added{border-left:4px solid #2f9e44}
.bb-removed{border-left:4px solid #e03131}
</style>""", unsafe_allow_html=True)


def chip(text, cls=""):
    return f'<span class="bb-chip {cls}">{text}</span>'


def short(text, n=110):
    text = (text or "").replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + "…"


def word_diff(old, new, limit=600):
    """(old_html, new_html) with removed words struck through on the left and new words marked on the right."""
    a, b = re.findall(r"\s+|\S+", (old or "")[:limit]), re.findall(r"\s+|\S+", (new or "")[:limit])
    left, right = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        x, y = html.escape("".join(a[i1:i2])), html.escape("".join(b[j1:j2]))
        if tag == "equal":
            left.append(x)
            right.append(y)
        else:
            if x:
                left.append(f'<span class="bb-del">{x}</span>')
            if y:
                right.append(f'<span class="bb-ins">{y}</span>')
    return "".join(left) or "∅ (empty)", "".join(right) or "∅ (empty)"


ss = st.session_state
ss.setdefault("selected_step", None)
ss.setdefault("replay_result", None)
ss.setdefault("diag_cache", {})
ss.setdefault("explain_cache", {})

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⬛ Black Box")
    st.caption("A flight recorder for AI agents")
    with st.expander("▶ Record a new run", expanded=False):
        t_opts = {f"{t.task_id}{' ★' if t.task_id in tasks.DEMO_TASK_IDS else ''} · {t.text}": t for t in tasks.TASKS}
        t_lbl = st.selectbox("Task", list(t_opts), key="rec_task")
        model = st.selectbox("Model", MODELS, key="rec_model", help=MODEL_HELP)
        model = st.text_input("…or another model id", "", key="rec_model_custom", help=MODEL_HELP,
                              placeholder="groq:<model>  openai:<model>  ollama:<model>").strip() or model
        framework = st.selectbox("Framework", ["custom", "langgraph"], key="rec_fw")
        ftype = st.selectbox("Inject fault", ["none"] + inject.FAULT_TYPES, key="rec_fault")
        fstep = st.number_input("Fault at step (0 = first compatible)", 0, 40, 0, key="rec_fstep")
        if st.button("Run agent", type="primary", key="rec_go"):
            fault = None if ftype == "none" else {"type": ftype, "step_no": int(fstep) or None, "seed": 1}
            with st.spinner("Running agent…"):
                try:
                    r = runner.record_run(t_opts[t_lbl], model, framework, fault=fault)
                    ss["run_pick"] = r["run_id"]
                    ss.replay_result, ss.selected_step = None, None
                    st.success(f"Recorded {r['run_id']}: {r['outcome']}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Run failed: {e}")

    flt = st.radio("Show runs", ["fail", "all", "success"], horizontal=True, key="flt")
    runs = store.list_runs(outcome=None if flt == "all" else flt)
    if not runs:
        st.info("No runs yet. Record one above or run `python run_batch.py`.")
        st.stop()
    labels = {r["run_id"]: f"{r['run_id']} · {'✅' if r['outcome'] == 'success' else '❌'} · {r['model']} · {short(r['task'], 40)}"
              for r in runs}
    ids = list(labels)
    if ss.get("run_pick") not in ids:
        ss["run_pick"] = ids[0]
    run_id = st.selectbox("Run", ids, format_func=labels.get, key="run_pick")
    st.caption(f"Diagnose API: `{clients.DIAGNOSE_URL}`  \nReplay API: `{clients.REPLAY_URL}`")

if ss.get("_last_run") != run_id:
    ss._last_run, ss.replay_result, ss.selected_step = run_id, None, None

run = store.get_run(run_id)
task = tasks.get_task(run.get("task_id") or "") or tasks.get_task_by_text(run["task"])

# ------------------------------------------------------------------ header
st.subheader(f"{run_id}: {run['task']}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Outcome", "✅ success" if run["outcome"] == "success" else "❌ fail")
c2.metric("Steps", len(run["steps"]))
c3.metric("Tokens", sum(s["tokens"] for s in run["steps"]))
c4.metric("Model", f"{run['model']}")
c5.metric("Framework", run["framework"])
st.caption(f"Final answer: **{run.get('final_answer', '?')}** · expected: **{tasks.fmt_value(task.expected) if task else '?'}**"
           + (f" · injected fault: `{run['fault']['type']}` @ step {run['fault']['step_no']} (label, hidden from the diagnosis)"
              if run.get("fault") else ""))

# ------------------------------------------------------------------ diagnose
if run_id not in ss.diag_cache:
    with st.spinner("Asking /diagnose…"):
        ss.diag_cache[run_id] = clients.diagnose(run)
diag, diag_src = ss.diag_cache[run_id]
suspects = {s["step_no"]: (i + 1, s) for i, s in enumerate(diag.get("suspects", []))}

left, right = st.columns([3, 2], gap="large")

# ------------------------------------------------------------------ right: suspects + edit
with right:
    st.markdown("### 🔎 Top suspects")
    st.caption(f"Source: {diag_src}")
    if st.button("↻ Re-diagnose", key="rediag"):
        ss.diag_cache.pop(run_id, None)
        st.rerun()
    for no, (rank, s) in suspects.items():
        with st.container(border=True):
            a, b = st.columns([3, 1])
            a.markdown(f"**#{rank} · step {no}** · {next(x['actor'] for x in run['steps'] if x['step_no'] == no)}")
            a.progress(float(s["confidence"]), text=f"confidence {s['confidence']:.0%}")
            a.markdown("".join(chip(e) for e in s["evidence"]), unsafe_allow_html=True)
            if b.button("Inspect", key=f"insp_{no}"):
                ss.selected_step = no
                st.rerun()

    st.markdown("### 🧠 Why did it fail?")
    ex_llm = st.text_input("Explain with (optional)", os.environ.get("BLACKBOX_EXPLAIN_LLM", ""), key="ex_llm",
                           placeholder="llama3  groq:<model>  openai:<model>  (empty = no LLM)",
                           help="The diagnosis model picks the suspect step. The LLM only words the explanation. " + MODEL_HELP)
    if st.button("Explain this failure", key="go_explain"):
        with st.spinner("Writing the explanation…"):
            ss.explain_cache[run_id] = clients.explain(run, ex_llm.strip() or None)
    ex = ss.explain_cache.get(run_id)
    if ex and ex.get("error"):
        st.warning(ex["error"])
    elif ex:
        with st.container(border=True):
            st.markdown(f"**{ex['summary']}**")
            st.markdown(f"**Root cause.** {ex['root_cause']}")
            st.markdown(f"**How it spread.** {ex['how_it_spread']}")
            st.markdown(f"**Suggested fix.** {ex['suggested_fix']}")
            if ex.get("other_suspects"):
                st.caption(ex["other_suspects"])
            st.caption(f"Explanation source: {ex['source']}")

    st.markdown("### ✏️ Edit & replay")
    step_nos = [s["step_no"] for s in run["steps"]]
    default = ss.selected_step or (next(iter(suspects)) if suspects else step_nos[0])
    sel = st.selectbox("Step", step_nos, index=step_nos.index(default) if default in step_nos else 0,
                       format_func=lambda n: f"{n} · {run['steps'][n - 1]['actor']} ({run['steps'][n - 1]['kind']})",
                       key=f"edit_step_{run_id}")
    ss.selected_step = sel
    stp = run["steps"][sel - 1]
    rw = replay.rewind(run, sel)
    st.caption(f"Rewound to just before step {sel}: {len(rw['before'])} earlier steps frozen · "
               f"an edit reruns {len(rw['will_rerun'])} dependent step(s) {rw['will_rerun']} · "
               f"{len(rw['will_reuse'])} reused")
    with st.expander(f"Restored state at this checkpoint · {rw['state']['state_hash']}", expanded=False):
        stt_ = rw["state"]
        st.caption(f"{stt_['steps_done']} steps done · {stt_['tokens_spent']} tokens already spent · "
                   "a replay starts from exactly this state without calling the model or tools again")
        if stt_["plan"]:
            st.markdown("**Plan in force**")
            st.json(stt_["plan"], expanded=False)
        for r_ in stt_["results_so_far"]:
            st.markdown(f"`#{r_['step_no']} {r_['actor']}` → {short(r_['output'], 90) or '∅ (empty)'}"
                        + (f"  ⚠ `{r_['error']}`" if r_["error"] else ""))
    mode = st.radio("Edit", ["Edit output", "Edit input", "Retry as-is", "Branch from here"], horizontal=True,
                    key=f"mode_{run_id}_{sel}")
    do_branch, alt_model = False, None
    if mode == "Edit output":
        val = st.text_area("New output", stp["output"], height=120, key=f"out_{run_id}_{sel}")
        edit = {"output": val}
        if val == stp["output"]:
            st.warning("Output unchanged: the replay will reproduce the same run.")
    elif mode == "Edit input":
        val = st.text_area("New input", stp["input"], height=160, key=f"in_{run_id}_{sel}")
        edit = {"input": val}
        if val == stp["input"]:
            st.warning("Input unchanged: this is the same as Retry as-is.")
    elif mode == "Retry as-is":
        edit = {"rerun": True}
    else:
        do_branch, edit = True, None
        st.caption(f"Alternative execution: steps before {sel} are restored from the record; step {sel} and "
                   f"all {len(run['steps']) - sel} step(s) after it run again, with the model you pick.")
        opts = [run["model"]] + [m for m in MODELS if m != run["model"]]
        alt_model = st.selectbox("Model from this step on", opts, key=f"alt_{run_id}_{sel}", help=MODEL_HELP)
        alt_model = st.text_input("…or another model id", "", key=f"altc_{run_id}_{sel}",
                                  placeholder="groq:<model>  openai:<model>  ollama:<model>").strip() or alt_model
    n_runs = st.slider("Replays (answers vary)", 1, 5, 3, key="nruns")
    if st.button("⏪ Replay from here", type="primary", key="go_replay"):
        with st.spinner("Running from the checkpoint…" if do_branch else "Smart replay…"):
            try:
                ss.replay_result = (*clients.replay(run_id, sel, edit, n_runs=n_runs, branch=do_branch,
                                                    model=alt_model), sel, edit)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

# ------------------------------------------------------------------ left: timeline
with left:
    st.markdown("### 🧭 Step timeline")
    dirty = set(rw["will_rerun"])
    for s in run["steps"]:
        no = s["step_no"]
        cls = "bb-card"
        if no in suspects:
            cls += " bb-sus"
        if no == sel:
            cls += " bb-sel"
        tags = [chip(s["kind"]), chip(f"{s['tokens']} tok")]
        if s["error"]:
            tags.append(chip(f"error: {s['error']}", "bb-err"))
        if no in suspects:
            r_, sus = suspects[no]
            tags.append(chip(f"SUSPECT #{r_} · {sus['confidence']:.0%}", "bb-rerun"))
        if no == sel:
            tags.append(chip("edit here", "bb-edited"))
        elif no in dirty:
            tags.append(chip("would rerun", "bb-rerun"))
        deps = f"← {s['depends_on']}" if s["depends_on"] else ""
        st.markdown(
            f'<div class="{cls}"><b>#{no} · {s["actor"]}</b> <span style="opacity:.6">{deps}</span><br>'
            f'{"".join(tags)}<div class="bb-mono">→ {short(s["output"], 160) or "∅ (empty)"}</div></div>',
            unsafe_allow_html=True)
        with st.expander(f"details · step {no}", expanded=False):
            st.markdown("**Input**")
            st.code(s["input"], language=None)
            st.markdown("**Output**")
            st.code(s["output"] or "(empty)", language=None)

# ------------------------------------------------------------------ results
res = ss.replay_result
if res:
    data, src, ed_step, ed = res
    new = data["new_run"]
    st.divider()
    branch_mode = data.get("mode") == "branch"
    st.markdown(f"## 🔁 {'Alternative run from checkpoint' if branch_mode else 'Replay result: edit at'} step {ed_step}")
    st.caption(f"via {src} · {data.get('n_runs', 1)} replay(s) · success rate {data.get('success_rate', 0):.0%}"
               + (f" · model from step {ed_step} on: `{data.get('model')}`" if branch_mode else ""))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Outcome", f"{run['outcome']} → {new['outcome']}",
              "flipped ✅" if data["outcome_flipped"] else "not flipped", delta_color="normal" if data["outcome_flipped"] else "off")
    m2.metric("Steps rerun", data["steps_rerun"])
    m3.metric("Steps reused", data["steps_reused"])
    m4.metric("Tokens saved", f"{data['tokens_saved_pct']}%")
    st.progress(min(1.0, data["tokens_saved_pct"] / 100), text=f"Savings meter: {data['tokens_saved_pct']}% of tokens not rerun vs a full rerun")
    rst = data.get("restore") or new.get("replay", {}).get("restore")
    if rst:
        if rst["exact"]:
            st.caption(f"✅ State restored exactly: all {rst['prefix_steps']} step(s) before step {ed_step} came from the "
                       f"record, none were called again · state `{rst['state_hash']}` = replayed `{rst['replayed_hash']}`")
        else:
            st.warning(f"State before step {ed_step} was not restored exactly: step(s) {rst['mismatched_steps']} "
                       "ran again or came back different.")

    cmp_ = data.get("comparison") or replay.compare(run, new)
    sm = cmp_["summary"]
    st.markdown(f"### Original vs modified trace · {run['run_id']} → {new['run_id']}")
    st.caption(f"{sm['same']} same · {sm['changed']} changed · {sm['added']} added · {sm['removed']} removed · "
               f"first difference at step {sm['first_divergence']} · steps {sm['steps'][0]} → {sm['steps'][1]} · "
               f"tokens {sm['tokens'][0]} → {sm['tokens'][1]}")
    only_diff = st.toggle("Show only steps that differ", value=False, key="only_diff")
    h1, h2 = st.columns(2, gap="large")
    h1.markdown(f"**Original · {run['outcome']}**  \n{run.get('final_answer', '')}")
    h2.markdown(f"**Modified · {new['outcome']}**  \n{new.get('final_answer', '')}")
    for row in cmp_["steps"]:
        if only_diff and row["status"] == "same":
            continue
        lo, ro = word_diff(row["a_output"], row["b_output"]) if row["status"] == "changed" else (
            html.escape(short(row["a_output"] or "", 300)) or "∅ (empty)", html.escape(short(row["b_output"] or "", 300)) or "∅ (empty)")
        tags = chip(row["status"], "bb-err" if row["status"] != "same" else "")
        if row["replay_status"]:
            tags += chip(row["replay_status"], "bb-" + row["replay_status"])
        if row["input_changed"]:
            tags += chip("input changed", "bb-rerun")
        if row["error_changed"]:
            tags += chip(f"error: {row['a_error']} → {row['b_error']}", "bb-err")
        o_col, n_col = st.columns(2, gap="large")
        if row["a_step"] is not None:
            o_col.markdown(f'<div class="bb-card bb-{row["status"]}"><b>#{row["a_step"]} {row["actor"]}</b>'
                           f'<div class="bb-mono">{lo}</div></div>', unsafe_allow_html=True)
        if row["b_step"] is not None:
            n_col.markdown(f'<div class="bb-card bb-{row["status"]}"><b>#{row["b_step"]} {row["actor"]}</b> {tags}'
                           f'<div class="bb-mono">{ro}</div></div>', unsafe_allow_html=True)
        if row["input_changed"]:
            with st.expander(f"input diff · step {row['b_step']}", expanded=False):
                li, ri = word_diff(row["a_input"], row["b_input"], limit=2000)
                d1, d2 = st.columns(2, gap="large")
                d1.markdown(f'<div class="bb-mono">{li}</div>', unsafe_allow_html=True)
                d2.markdown(f'<div class="bb-mono">{ri}</div>', unsafe_allow_html=True)

    if data["outcome_flipped"] and new["outcome"] == "success":
        st.success(f"Step {ed_step} is confirmed as the cause: fixing it flips the run to success.")
        if st.button("💾 Save as confirmed cause (training data + regression test)", key="confirm"):
            CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
            p = CONFIRMED_DIR / f"{run['run_id']}_step{ed_step}.json"
            p.write_text(json.dumps({"run_id": run["run_id"], "confirmed_step": ed_step, "edit": ed or {"branch": data.get("model")},
                                     "fixed_run_id": new["run_id"], "suspects": diag.get("suspects", []),
                                     "run": run}, indent=2), encoding="utf-8")
            st.toast(f"Saved {p.name}")
