"""Accuracy report: baseline model vs improved model, simulated vs real, per fault type.

    python -m part2.eval
    python -m part2.eval --real-dir "blackbox_part 1/runs"   # adds Part 1's recorded runs
    python -m part2.eval --real-dir ... --confirmed          # before/after adding part2/data/confirmed.jsonl
    python -m part2.eval --real-dir ... --llm llama3 --llm-n 40      # LLM-judge baseline (Ollama)
    python -m part2.eval --real-dir ... --llm groq:<model>           # same with a hosted model

Three pipelines are compared on the same runs:
  baseline      the previous feature set, model score only
  new signals   the current feature set, model score only
  final         the current feature set + decisive-evidence reranking (what /diagnose serves)

Data roles (never mixed):
  simulated train     seen-setup simulated runs, 4/5 of the source runs
  simulated held-out  the other 1/5 (seen setup) and every crewai / mistral run (unseen setup)
  real train          Part 1 runs on training tasks, seen setups
  validation          grouped 5-fold cross-validation inside real train (whole tasks per fold);
                      features, reranker and threshold were chosen here
  real held-out       Part 1 runs on the other half of the tasks; only scored, never trained on
"""
import argparse
import os
import random
from collections import defaultdict

from . import baseline, model, validate
from .data import (DATA, HELD_FRAMEWORK, HELD_MODEL, ROOT, confirmed_for_training, cv_fold, fault_of, group_key,
                   is_real_llm, is_unseen, read_jsonl, real_parts, split_twins)
from .signals import BASELINE_FEATURES, EVIDENCE, extract

HEUR = "Heuristic (first error, else last step)"


class Pipe:
    """A way of ranking steps: a trained booster plus whether decisive evidence reranks."""

    def __init__(self, name, booster, rerank):
        self.name, self.booster, self.rerank = name, booster, rerank

    def rank(self, run):
        return model.rank(self.booster, run, rerank=self.rerank)

    def scores(self, run):
        return model.final_scores(self.booster, run, rerank=self.rerank)


BASE, NEW, FINAL = "Baseline (previous signals)", "New signals, no reranker", "Black Box model"


def pipes(train_runs, tag=""):
    """(baseline, new signals, final) trained on the same runs. The final pipeline keeps the
    name "Black Box model" so the results page of the UI finds it."""
    old = model.train(train_runs, features=BASELINE_FEATURES)
    new = model.train(train_runs)
    return [Pipe(BASE, old, False), Pipe(NEW, new, False), Pipe(FINAL + tag, new, True)]


def score(runs, rank):
    t1 = t3 = 0
    for run in runs:
        top = rank(run)
        t1 += bool(top) and top[0] == run["faulty_step"]
        t3 += run["faulty_step"] in top
    n = max(1, len(runs))
    return 100 * t1 / n, 100 * t3 / n


def prf(pairs, threshold):
    """pairs: [(scores, truths)] per run. Each step is one yes/no decision."""
    tp = fp = fn = tn = 0
    for s, y in pairs:
        for p, truth in zip(s, y):
            flag = p >= threshold
            tp += truth and flag
            fp += flag and not truth
            fn += truth and not flag
            tn += not truth and not flag
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def pairs_of(runs, pipe):
    return [(pipe.scores(r), [s["step_no"] == r["faulty_step"] for s in r["steps"]]) for r in runs]


def validation(sim_train, real_train, k=5):
    """Grouped cross-validation over the real training tasks. Returns per-pipeline
    out-of-fold (run, rank, scores) so accuracy and thresholds are measured on runs the
    fold's model never saw."""
    out = defaultdict(list)
    for f in range(k):
        fit = [r for r in real_train if cv_fold(r, k) != f]
        held = [r for r in real_train if cv_fold(r, k) == f]
        if not held:
            continue
        for p in pipes(sim_train + fit):
            for r in held:
                out[p.name].append((r, p.rank(r), p.scores(r)))
    return out


def fault_rows(runs, columns):
    """columns: [(header, rank fn)]. One row per fault type with top-1 / top-3 for each column."""
    by_fault = defaultdict(list)
    for r in runs:
        by_fault[fault_of(r)].append(r)
    head = "| Fault | Runs | " + " | ".join(f"{h} top-1 | {h} top-3" for h, _ in columns) + " |"
    out = [head, "|---|---|" + "---|---|" * len(columns)]
    for ft, group in sorted(by_fault.items()) + [("all", runs)]:
        cells = []
        for _, rank in columns:
            t1, t3 = score(group, rank)
            cells += [f"{t1:.1f}%", f"{t3:.1f}%"]
        out.append(f"| {ft} | {len(group)} | " + " | ".join(cells) + " |")
    return out


def kind_matrix(title, runs, rank):
    """Where the top suspect lands: kind of the truly faulty step vs kind of the step ranked first."""
    kinds = ["model_call", "tool_call", "decision"]
    grid = defaultdict(int)
    for run in runs:
        by_no = {s["step_no"]: s.get("kind") for s in run["steps"]}
        top = rank(run)
        if top:
            hit = "right step" if top[0] == run["faulty_step"] else "wrong step, " + str(by_no.get(top[0]))
            grid[(by_no.get(run["faulty_step"]), hit)] += 1
    cols = ["right step"] + ["wrong step, " + k for k in kinds]
    out = ["", f"### {title}", "", "Rows: kind of the truly faulty step. Columns: what was ranked first.", "",
           "| Faulty step is a | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k in kinds:
        if any(grid[(k, c)] for c in cols):
            out.append(f"| {k} | " + " | ".join(str(grid[(k, c)]) for c in cols) + " |")
    return out


def unseen_failure_section(train_runs, tests, full):
    """Leave one fault type out of training entirely, then test only on that fault type."""
    out = ["", "## Unseen failure types (final pipeline)", "",
           "For each row the model is retrained with that fault type removed from training, then tested only on "
           "runs with that fault. \"Seen\" is the normal model, which had examples of it.", "",
           "| Fault type held out | Test set | Runs | Top-1 never seen | Top-3 never seen | Top-1 seen | Top-3 seen |",
           "|---|---|---|---|---|---|---|"]
    totals = defaultdict(lambda: [0, 0, 0, 0, 0])
    for ft in sorted({fault_of(r) for r in train_runs}):
        blind = Pipe("", model.train([r for r in train_runs if fault_of(r) != ft]), True)
        for name, runs in tests:
            group = [r for r in runs if fault_of(r) == ft]
            if not group:
                continue
            n1, n3 = score(group, blind.rank)
            s1, s3 = score(group, full.rank)
            out.append(f"| {ft} | {name} | {len(group)} | {n1:.1f}% | {n3:.1f}% | {s1:.1f}% | {s3:.1f}% |")
            t = totals[name]
            for i, v in enumerate((n1, n3, s1, s3)):
                t[i] += v * len(group)
            t[4] += len(group)
    for name, t in totals.items():
        n = max(1, t[4])
        out.append(f"| **all, each held out in turn** | {name} | {t[4]} | {t[0] / n:.1f}% | {t[1] / n:.1f}% | "
                   f"{t[2] / n:.1f}% | {t[3] / n:.1f}% |")
    return out


def fire_rates(runs):
    """Share of faulty steps, and of healthy steps, on which each evidence signal fires."""
    hit_f, hit_h, nf, nh = defaultdict(int), defaultdict(int), 0, 0
    for run in runs:
        for step, row in zip(run["steps"], extract(run)):
            faulty = step["step_no"] == run["faulty_step"]
            nf += faulty
            nh += not faulty
            for _, label, test in EVIDENCE:
                if test(row):
                    (hit_f if faulty else hit_h)[label] += 1
    return {label: (100 * hit_f[label] / max(1, nf), 100 * hit_h[label] / max(1, nh)) for _, label, _ in EVIDENCE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm")
    ap.add_argument("--llm-n", type=int, default=40)
    ap.add_argument("--real-dir")
    ap.add_argument("--confirmed", action="store_true")
    a = ap.parse_args()

    failed = read_jsonl(os.path.join(DATA, "failed.jsonl"))
    seen = [r for r in failed if not is_unseen(r)]
    unseen = [r for r in failed if is_unseen(r)]
    # split seen runs by the good run they came from, so siblings never straddle train/test
    sources = sorted({r["source_run_id"] for r in seen})
    random.Random(1).shuffle(sources)
    test_src = set(sources[: len(sources) // 5])
    train = [r for r in seen if r["source_run_id"] not in test_src]
    test = [r for r in seen if r["source_run_id"] in test_src]
    sim = pipes(train)
    heur = (HEUR, baseline.heuristic)

    def methods(ps):
        return [(p.name, p.rank) for p in reversed(ps)] + [heur]

    # (test set, training setup, runs, [(method, rank fn)])
    sim_setup = f"{len(train)} simulated"
    sets = [("SIMULATED, seen setup, held-out source runs", sim_setup, test, methods(sim)),
            (f"SIMULATED, unseen setup ({HELD_FRAMEWORK} or {HELD_MODEL})", sim_setup, unseen, methods(sim))]
    real, real_train, heldout, both, llm_only = [], [], [], None, []
    if a.real_dir:
        real = validate.labelled_failures(a.real_dir)
        real_train, real_seen, real_unseen = real_parts(real)
        heldout, twins = split_twins(real_seen + real_unseen)
        llm_only = [r for r in heldout if is_real_llm(r)]
        everything, _ = split_twins(real)
        zero = [Pipe(p.name + (" (simulated only)" if p.name == FINAL else ""), p.booster, p.rerank) for p in sim]
        sets.append(("REAL, every labelled run (twins removed)", sim_setup + ", no real runs", everything, methods(zero)))
        if real_train:
            both = pipes(train + real_train, f" (simulated + {len(real_train)} real)")
            setup = f"{len(train)} simulated + {len(real_train)} real"
            sets.append(("REAL, seen setup, held-out tasks (twins removed)", setup, heldout, methods(both)))
            if llm_only:
                names = ", ".join(sorted({r["model"] for r in llm_only}))
                sets.append((f"REAL LLM runs only ({names}), held-out tasks", setup, llm_only, methods(both)))
                for m in sorted({r["model"] for r in llm_only}):
                    fit = [r for r in real_train if r["model"] != m]
                    sets.append((f"REAL, unseen model: `{m}` removed from training, held-out tasks",
                                 f"{len(train)} simulated + {len(fit)} real (none from this model)",
                                 [r for r in llm_only if r["model"] == m],
                                 methods(pipes(train + fit, f" (simulated + {len(fit)} real)"))))
            if twins:
                sets.append(("REAL, twin runs (LangGraph copies of runs above, not independent)", setup, twins,
                             [(both[2].name, both[2].rank)]))
    final_sim, final_real = sim[2], both[2] if both else None
    short = lambda p: FINAL if p.name.startswith(FINAL) else p.name  # noqa: E731

    out = ["# Black Box diagnosis results", "",
           f"Three pipelines on the same runs: **{BASE}** (the 25 earlier signals, model score only), **{NEW}** "
           f"(all 32 signals, model score only), and **{FINAL}** (32 signals + decisive-evidence reranker; this is "
           "what `/diagnose` serves).", "",
           f"Simulated runs come from `sim_agent.py`; framework `{HELD_FRAMEWORK}` and model `{HELD_MODEL}` are held "
           "out entirely and train and test never share a source run. "
           + (f"REAL runs come from Part 1's recorder (`{a.real_dir}`, models: "
              f"{', '.join(sorted({r['model'] for r in real}))}). Real runs are split by task: {len(real_train)} runs on "
              f"training tasks, {len(heldout)} independent runs on held-out tasks. Features, reranker and threshold "
              "were chosen by cross-validation inside the training tasks; held-out tasks are only scored."
              if a.real_dir else "No real runs are in this report."), "",
           "| Test set | Runs | Method | Top-1 | Top-3 | Trained on |", "|---|---|---|---|---|---|"]
    for set_name, setup, runs, ms in sets:
        if not runs:
            continue
        for name, rank in ms:
            t1, t3 = score(runs, rank)
            out.append(f"| {set_name} | {len(runs)} | {name} | {t1:.1f}% | {t3:.1f}% | {'-' if name == HEUR else setup} |")
        if a.llm:
            sample = random.Random(2).sample(runs, min(a.llm_n, len(runs)))
            failures = []
            t1, t3 = score(sample, lambda run: baseline.llm_judge(run, a.llm, failures=failures))
            ours = ms[0]
            m1, m3 = score(sample, ours[1])
            note = f", {len(failures)} calls gave no usable answer" if failures else ""
            out.append(f"| {set_name} | {len(sample)} | LLM judge ({a.llm}{note}), sample | {t1:.1f}% | {t3:.1f}% | - |")
            out.append(f"| {set_name} | {len(sample)} | {ours[0]}, same sample | {m1:.1f}% | {m3:.1f}% | {setup} |")

    if both:
        oof = validation(train, real_train)
        oof = {BASE: oof[BASE], NEW: oof[NEW], FINAL: next(v for k, v in oof.items() if k.startswith(FINAL))}
        n = len(oof[BASE])
        out += ["", "## Validation (how the choices were made)", "",
                f"Grouped 5-fold cross-validation over the {len(real_train)} real training runs: each fold holds out whole "
                f"tasks, the model is trained on the simulated runs plus the other folds. {n} out-of-fold runs.", "",
                "| Pipeline | Runs | Top-1 | Top-3 |", "|---|---|---|---|"]
        for name, rows in oof.items():
            t1 = 100 * sum(top[0] == r["faulty_step"] for r, top, _ in rows) / n
            t3 = 100 * sum(r["faulty_step"] in top for r, top, _ in rows) / n
            out.append(f"| {name} | {n} | {t1:.1f}% | {t3:.1f}% |")
        out += ["", "Flag threshold, on the same out-of-fold scores (a step is flagged when its score reaches the threshold):", "",
                "| Threshold | Baseline P | Baseline R | Baseline F1 | Final P | Final R | Final F1 |", "|---|---|---|---|---|---|---|"]
        for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            cells = []
            for name in (BASE, FINAL):
                m = prf([(s, [st["step_no"] == r["faulty_step"] for st in r["steps"]]) for r, _, s in oof[name]], t)
                cells += [f"{100 * m['precision']:.1f}%", f"{100 * m['recall']:.1f}%", f"{100 * m['f1']:.1f}%"]
            mark = " (chosen)" if abs(t - model.THRESHOLD) < 1e-9 else " (old)" if t == 0.5 else ""
            out.append(f"| {t}{mark} | " + " | ".join(cells) + " |")
        out += ["", f"Chosen threshold: {model.THRESHOLD}. The final pipeline's F1 is flat between 0.7 and 0.9 on validation, "
                "so the middle of that range was taken instead of the single best value."]

    named = [("SIMULATED, unseen setup", unseen, sim[0], final_sim)]
    if both:
        named.append(("REAL, held-out tasks", heldout, both[0], final_real))
        if llm_only:
            named.append(("REAL LLM runs only, held-out tasks", llm_only, both[0], final_real))
    out += ["", "## Precision, recall, F1 and confusion matrix (per step, held-out data)", "",
            "Every step is one decision. Each run has exactly one truly faulty step.", "",
            "| Test set | Steps | Precision | Recall | F1 | Pipeline | Threshold |", "|---|---|---|---|---|---|---|"]
    mats = []
    for name, runs, old, new in named:
        for pipe, thr in ((new, model.THRESHOLD), (new, 0.5), (old, 0.5)):
            m = prf(pairs_of(runs, pipe), thr)
            total = m["tp"] + m["fp"] + m["fn"] + m["tn"]
            out.append(f"| {name} | {total} | {100 * m['precision']:.1f}% | {100 * m['recall']:.1f}% | "
                       f"{100 * m['f1']:.1f}% | {short(pipe)} | {thr} |")
            if (pipe is old) or thr == model.THRESHOLD:
                mats.append((f"{name}, {short(pipe)}, threshold {thr}", m))
    for title, m in mats:
        out += ["", f"Confusion matrix: {title}", "",
                "| | Flagged faulty | Not flagged |", "|---|---|---|",
                f"| Truly faulty step | {m['tp']} (TP) | {m['fn']} (FN) |",
                f"| Healthy step | {m['fp']} (FP) | {m['tn']} (TN) |"]

    out += ["", "## By fault type", "", f"### SIMULATED unseen setup ({len(unseen)} runs, trained on {sim_setup})", ""]
    out += fault_rows(unseen, [("Baseline", sim[0].rank), ("Final", final_sim.rank)])
    if both:
        out += ["", f"### REAL held-out tasks ({len(heldout)} runs, trained on {len(train)} simulated + {len(real_train)} real)", ""]
        out += fault_rows(heldout, [("Baseline", both[0].rank), ("New signals", both[1].rank), ("Final", final_real.rank)])
        if llm_only:
            out += ["", f"### REAL LLM runs only, held-out tasks ({len(llm_only)} runs, same training)", ""]
            out += fault_rows(llm_only, [("Baseline", both[0].rank), ("Final", final_real.rank)])
        out += ["", f"### REAL held-out tasks, simulated-only training ({len(heldout)} runs, trained on {sim_setup}, no real runs)", ""]
        out += fault_rows(heldout, [("Baseline", sim[0].rank), ("New signals", sim[1].rank), ("Final", final_sim.rank)])

    out += ["", "## Where the top suspect lands (final pipeline)"]
    out += kind_matrix("SIMULATED unseen setup", unseen, final_sim.rank)
    if both:
        out += kind_matrix("REAL held-out tasks", heldout, final_real.rank)

    tests = [("SIMULATED, unseen setup", unseen)] + ([("REAL, held-out tasks", heldout)] if both else [])
    out += unseen_failure_section(train + real_train, tests, final_real or final_sim)

    if real:
        sim_rate, real_rate = fire_rates(unseen), fire_rates(real_train)
        out += ["", "## How often each signal fires", "",
                "On faulty steps / on healthy steps. Real numbers are from the training tasks only.", "",
                "| Signal | Simulated faulty | Simulated healthy | Real faulty | Real healthy |", "|---|---|---|---|---|"]
        out += [f"| {label} | {sim_rate[label][0]:.0f}% | {sim_rate[label][1]:.0f}% | {real_rate[label][0]:.0f}% | "
                f"{real_rate[label][1]:.0f}% |" for label in sim_rate]

    if a.confirmed:
        conf = confirmed_for_training(read_jsonl(os.path.join(DATA, "confirmed.jsonl")))
        used = {group_key(r) for r in conf}
        hold = [r for r in unseen + heldout if group_key(r) not in used]
        after = Pipe("", model.train(train + [{**r, "faulty_step": r["confirmed_step"]} for r in conf]), True)
        b1, b3 = score(hold, final_sim.rank)
        a1, a3 = score(hold, after.rank)
        out += ["", "## Learning from confirmed runs (final pipeline)", "",
                f"{len(conf)} runs confirmed by the verify loop were added to a simulated-only model, labelled with the "
                "step replay confirmed (not the injected label). Tested on held-out runs (simulated unseen setup, plus "
                "real held-out runs when given) whose task or source run is not among the added ones.", "",
                "| Runs | Trained on | Top-1 | Top-3 |", "|---|---|---|---|",
                f"| {len(hold)} | {sim_setup} | {b1:.1f}% | {b3:.1f}% |",
                f"| {len(hold)} | {sim_setup} + {len(conf)} replay-confirmed | {a1:.1f}% | {a3:.1f}% |"]

    text = "\n".join(out) + "\n"
    with open(os.path.join(ROOT, "part2", "results.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
