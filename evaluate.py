#!/usr/bin/env python3
"""Compute spec §6 metrics for a saved run.

  python evaluate.py runs/eval-2026-11-08                      # metrics + report.md
  python evaluate.py --compare runs/eval-A runs/eval-B          # §1 step 6: compare runs

Automatic: Hit@5 (doc / article), MRR, context precision (doc proxy), no-answer accuracy,
fabricated / outdated citations, quote grounding, latency p50/p95, error rate.
Manual (grading.csv, 0/1/2 + 0/1 columns): answer correctness, faithfulness, citation accuracy,
error cause (retrieval / generatsiya / manba).
"""
import argparse
import csv
import glob
import json
import math
import os

from common import article_num, norm_text

ACTIVE = {"amalda", "amaldagi", "действует", "действующий", "in_force", "active"}
TARGETS = [  # key, layer, label, op, target
    ("hit5_doc", "Retrieval", "Hit@5 (hujjat)", ">=", 0.90),
    ("hit5_article", "Retrieval", "Hit@5 (modda)", ">=", 0.80),
    ("mrr", "Retrieval", "MRR", ">=", 0.70),
    ("context_precision", "Retrieval", "Context precision (proxy: hujjat)", ">=", 0.60),
    ("answer_correctness", "Generatsiya", "Answer correctness 0-2 (qo'lda)", ">=", 1.6),
    ("faithfulness", "Generatsiya", "Faithfulness (qo'lda)", ">=", 0.95),
    ("quote_grounded", "Generatsiya", "Quote contexts ichida (avto)", ">=", 0.95),
    ("citation_accuracy", "Manba", "Citation accuracy (qo'lda)", ">=", 0.90),
    ("fabricated_citations", "Manba", "To'qilgan manba", "==", 0),
    ("outdated_citations", "Manba", "Eskirgan norma", "==", 0),
    ("no_answer_accuracy", "Xavfsizlik", "No-answer aniqligi", ">=", 0.90),
    ("false_no_answer", "Xavfsizlik", "Javobi bor savolga no_answer (info)", "info", None),
    ("latency_p50_ms", "Operatsion", "Latency p50, ms", "<=", 5000),
    ("latency_p95_ms", "Operatsion", "Latency p95, ms", "<=", 12000),
    ("error_rate", "Operatsion", "Xato ulushi", "<=", 0.02),
]
GRADE_COLS = ["correctness_0_1_2", "faithful_0_1", "citation_ok_0_1", "error_cause"]


# ---------- loading ----------
def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_run(run_dir):
    results, systems = {}, []
    for f in sorted(glob.glob(os.path.join(run_dir, "b*.json"))):
        if f.endswith((".request.json", ".problems.json")):
            continue
        with open(f, encoding="utf-8") as fh:
            resp = json.load(fh)
        systems.append(resp.get("system") or {})
        for r in resp.get("results") or []:
            results[r.get("id")] = r
    return results, systems


def expected(row, sources):
    """List of {key, doc_id, article}. expected_doc_key / _id / _article are '|'-aligned."""
    split = lambda s: [x.strip() for x in (s or "").split("|")]
    keys, ids, arts = split(row.get("expected_doc_key")), split(row.get("expected_doc_id")), split(row.get("expected_article"))
    n = max(len([k for k in keys if k]), len([i for i in ids if i]), len([x for x in arts if x]))
    out = []
    for i in range(n):
        k = keys[i] if i < len(keys) else ""
        d = ids[i] if i < len(ids) and ids[i] else (sources.get(k, {}).get("lex_id") or "")
        if d.upper().startswith("TODO"):
            d = ""
        out.append({"key": k, "doc_id": d, "article": article_num(arts[i]) if i < len(arts) else None})
    return out


# ---------- math ----------
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def percentile(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]


def passes(op, val, tgt):
    if val is None or op == "info":
        return None
    return {">=": val >= tgt, "<=": val <= tgt, "==": val == tgt}[op]


# ---------- core ----------
def evaluate_run(run_dir, testset, sources, write=True):
    rows = load_csv(testset)
    srcs = json.load(open(sources, encoding="utf-8")) if os.path.exists(sources) else {}
    results, systems = load_run(run_dir)

    per_q, lat = [], []
    fabricated = outdated = cits_total = grounded = 0
    for row in rows:
        qid, r = row["id"], results.get(row["id"])
        answerable = row.get("answerable", "true").strip().lower() == "true"
        q = {"id": qid, "category": row.get("category"), "answerable": answerable,
             "status": r.get("status") if r else "missing", "flags": []}
        per_q.append(q)
        if not r or r.get("status") in ("error", "timeout"):
            q["flags"].append(f"status={q['status']}")
            continue
        if (r.get("timing") or {}).get("total_ms") is not None:
            lat.append(r["timing"]["total_ms"])
        said_no = bool(r.get("no_answer")) or r.get("status") == "no_answer"
        q["no_answer"] = said_no

        ctx = sorted(r.get("contexts") or [], key=lambda c: c.get("rank", 999))
        top5 = ctx[:5]
        ctx_by_id = {c.get("chunk_id"): c for c in ctx}

        # --- source checks (all answered questions)
        for cit in r.get("citations") or []:
            cits_total += 1
            c = ctx_by_id.get(cit.get("chunk_id"))
            if c is None or "lex.uz" not in (cit.get("url") or ""):
                fabricated += 1
                q["flags"].append(f"fabricated:{cit.get('chunk_id')}")
            elif norm_text(cit.get("quote")) and norm_text(cit.get("quote")) in norm_text(c.get("text")):
                grounded += 1
            else:
                q["flags"].append(f"quote_not_in_context:ref{cit.get('ref')}")
            if norm_text(cit.get("doc_status")) not in ACTIVE:
                outdated += 1
                q["flags"].append(f"outdated:{cit.get('doc_id')}:{cit.get('doc_status')}")

        if not answerable:
            q["no_answer_ok"] = said_no
            if not said_no:
                q["flags"].append("hallucinated_answer_to_unanswerable")
            continue
        if said_no:
            q["flags"].append("false_no_answer")

        # --- retrieval checks (answerable, ground truth known)
        exp = expected(row, srcs)
        docs = {e["doc_id"] for e in exp if e["doc_id"]}
        arts = [e for e in exp if e["article"]]
        if docs:
            q["hit5_doc"] = any(c.get("doc_id") in docs for c in top5)
            q["doc_recall5"] = len(docs & {c.get("doc_id") for c in top5}) / len(docs)
            q["context_precision"] = (sum(c.get("doc_id") in docs for c in top5) / len(top5)) if top5 else 0.0
        if arts:
            def is_hit(c):
                return any(article_num(c.get("article")) == e["article"]
                           and (not e["doc_id"] or c.get("doc_id") == e["doc_id"]) for e in arts)
            q["hit5_article"] = any(is_hit(c) for c in top5)
            first = next((i for i, c in enumerate(ctx, 1) if is_hit(c)), None)
            q["rr"] = 1 / first if first else 0.0
            if not q["hit5_article"]:
                q["auto_cause"] = "retrieval"

    ans = [q for q in per_q if q["answerable"]]
    unans = [q for q in per_q if not q["answerable"]]
    m = {
        "hit5_doc": mean([q.get("hit5_doc") for q in ans]),
        "hit5_article": mean([q.get("hit5_article") for q in ans]),
        "mrr": mean([q.get("rr") for q in ans]),
        "context_precision": mean([q.get("context_precision") for q in ans]),
        "quote_grounded": grounded / cits_total if cits_total else None,
        "fabricated_citations": fabricated,
        "outdated_citations": outdated,
        "no_answer_accuracy": mean([q.get("no_answer_ok") for q in unans]),
        "false_no_answer": sum(1 for q in ans if q.get("no_answer")),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "error_rate": sum(q["status"] in ("error", "timeout", "missing") for q in per_q) / len(per_q),
    }
    coverage = {
        "questions": len(per_q), "results_found": sum(q["status"] != "missing" for q in per_q),
        "with_doc_truth": sum("hit5_doc" in q for q in ans),
        "with_article_truth": sum("hit5_article" in q for q in ans),
    }

    # --- manual grading sheet
    gpath = os.path.join(run_dir, "grading.csv")
    graded = {}
    if os.path.exists(gpath):
        for g in load_csv(gpath):
            graded[g["id"]] = g
    toi = lambda s: int(s) if str(s).strip().isdigit() else None
    m["answer_correctness"] = mean([toi(g.get("correctness_0_1_2")) for g in graded.values()])
    m["faithfulness"] = mean([toi(g.get("faithful_0_1")) for g in graded.values()])
    m["citation_accuracy"] = mean([toi(g.get("citation_ok_0_1")) for g in graded.values()])
    causes = {}
    for g in graded.values():
        c = (g.get("error_cause") or "").strip()
        if c:
            causes[c] = causes.get(c, 0) + 1
    coverage["manually_graded"] = sum(toi(g.get("correctness_0_1_2")) is not None for g in graded.values())

    out = {"run_dir": run_dir, "system": systems[0] if systems else {},
           "system_consistent": all(s == systems[0] for s in systems) if systems else None,
           "metrics": m, "coverage": coverage, "error_causes": causes,
           "per_category": per_category(per_q), "per_question": per_q}
    if write:
        write_outputs(run_dir, out, rows, results, per_q, gpath)
    return out


def per_category(per_q):
    cats = {}
    for q in per_q:
        cats.setdefault(q["category"], []).append(q)
    return {c: {"n": len(qs),
                "hit5_article": mean([q.get("hit5_article") for q in qs]),
                "hit5_doc": mean([q.get("hit5_doc") for q in qs]),
                "errors": sum(q["status"] in ("error", "timeout", "missing") for q in qs),
                "flagged": sum(bool(q["flags"]) for q in qs)} for c, qs in cats.items()}


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_outputs(run_dir, out, rows, results, per_q, gpath):
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if not os.path.exists(gpath):  # create sheet once; never overwrite human grades
        pq = {q["id"]: q for q in per_q}
        with open(gpath, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "category", "question", "expected_answer", "answer", "citations",
                         "auto_flags", "auto_cause"] + GRADE_COLS)
            for row in rows:
                r = results.get(row["id"]) or {}
                cits = "; ".join(f"[{c.get('ref')}] {c.get('doc_title', c.get('doc_id'))} {c.get('article')}"
                                 for c in r.get("citations") or [])
                q = pq[row["id"]]
                w.writerow([row["id"], row.get("category"), row["question"], row.get("expected_answer"),
                            r.get("answer"), cits, " ".join(q["flags"]), q.get("auto_cause", ""),
                            "", "", "", q.get("auto_cause", "")])

    m, cov = out["metrics"], out["coverage"]
    L = [f"# RAG eval report — `{run_dir}`", "",
         "**System:** " + ", ".join(f"{k}={v}" for k, v in out["system"].items()),
         f"(consistent across batches: {out['system_consistent']})", "",
         f"**Coverage:** {cov}", "",
         "| Qatlam | Metrika | Qiymat | Maqsad | Natija |", "|---|---|---|---|---|"]
    for key, layer, label, op, tgt in TARGETS:
        ok = passes(op, m.get(key), tgt)
        L.append(f"| {layer} | {label} | {fmt(m.get(key))} | {'' if op == 'info' else f'{op} {tgt}'} | "
                 f"{'—' if ok is None else ('✅' if ok else '❌')} |")
    L += ["", "## Kategoriya bo'yicha", "", "| Kategoriya | n | Hit@5 hujjat | Hit@5 modda | Xato | Flag |",
          "|---|---|---|---|---|---|"]
    for c, v in out["per_category"].items():
        L.append(f"| {c} | {v['n']} | {fmt(v['hit5_doc'])} | {fmt(v['hit5_article'])} | {v['errors']} | {v['flagged']} |")
    if out["error_causes"]:
        L += ["", "## Xato sabablari (grading.csv)", ""] + [f"- {k}: {v}" for k, v in out["error_causes"].items()]
    flagged = [q for q in per_q if q["flags"]]
    if flagged:
        L += ["", "## Flaglangan savollar", ""] + [f"- `{q['id']}` ({q['category']}): {', '.join(q['flags'])}" for q in flagged]
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:len(TARGETS) + 9]))
    print(f"\nwritten: {run_dir}/report.md, metrics.json, grading.csv")


def compare(a_dir, b_dir, testset, sources):
    A, B = (evaluate_run(d, testset, sources, write=False) for d in (a_dir, b_dir))
    print(f"{'metric':<34}{'A':>10}{'B':>10}{'delta':>10}")
    for key, _, label, _, _ in TARGETS:
        a, b = A["metrics"].get(key), B["metrics"].get(key)
        d = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        print(f"{label[:33]:<34}{fmt(a):>10}{fmt(b):>10}{fmt(d):>10}")
    print("\nsystem changes:")
    for k in sorted(set(A["system"]) | set(B["system"])):
        if A["system"].get(k) != B["system"].get(k):
            print(f"  {k}: {A['system'].get(k)} -> {B['system'].get(k)}")
    qa = {q["id"]: q for q in A["per_question"]}
    print("\nper-question changes (Hit@5 modda / no_answer):")
    for qb in B["per_question"]:
        q = qa.get(qb["id"], {})
        for k in ("hit5_article", "no_answer_ok"):
            if k in q and k in qb and q[k] != qb[k]:
                print(f"  {qb['id']} {k}: {q[k]} -> {qb[k]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--testset", default="testset_uyjoy.csv")
    ap.add_argument("--sources", default="sources.json")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare, a.testset, a.sources)
    elif a.run_dir:
        evaluate_run(a.run_dir, a.testset, a.sources)
    else:
        ap.error("give a run_dir or --compare A B")


if __name__ == "__main__":
    main()
