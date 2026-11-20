#!/usr/bin/env python3
"""Automates the spec §7 developer checklist against a live API.

  python contract_check.py --base-url https://... --api-key ...
Exit code 0 = all checks passed.
"""
import argparse
import copy
import os
import sys
import time

from common import http, validate_batch_response
from run_eval import poll

QUESTIONS = [  # housing / communal; mixes ‘ and ʻ apostrophes, Cyrillic and Russian
    {"id": "c1", "question": "Ko‘p kvartirali uyda umumiy mulkka nimalar kiradi?", "language": "uz-Latn"},
    {"id": "c2", "question": "Shirkat umumiy yigʻilishi qanday chaqiriladi?", "language": "uz-Latn"},
    {"id": "c3", "question": "Коммунал хизматлар учун тўловни ким тўлаши керак?", "language": "uz-Cyrl"},
    {"id": "c4", "question": "Кто несёт расходы на содержание общего имущества многоквартирного дома?", "language": "ru"},
    {"id": "c5", "question": "Marsda dacha qurish uchun qanday ruxsatnoma olinadi?", "language": "uz-Latn"},  # unanswerable
]
UNANSWERABLE = "c5"
MOJIBAKE = ("\ufffd", "Ã", "Ð", "Ñ\x80", "â€")

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))


def body(run_id, batch_id, items):
    return {"run_id": run_id, "batch_id": batch_id, "batch_index": 1, "total_batches": 1,
            "options": {"top_k": 5, "return_contexts": True, "return_debug": True,
                        "temperature": 0, "seed": 42, "answer_language": "auto"},
            "items": items}


def post(base, key, b):
    t0 = time.time()
    code, h, data = http("POST", f"{base}/v1/rag/batch", key, b)
    el = time.time() - t0
    if code == 202:
        data = poll(base, key, b["batch_id"])
        code = 200
    return code, data, el


def sig(resp):
    """Sources signature for determinism: per id -> (citation chunks, context chunks)."""
    return {r["id"]: (tuple(c.get("chunk_id") for c in r.get("citations") or []),
                      tuple(c.get("chunk_id") for c in r.get("contexts") or []))
            for r in resp.get("results") or []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("RAG_BASE_URL", "http://localhost:8080"))
    ap.add_argument("--api-key", default=os.environ.get("RAG_API_KEY"))
    a = ap.parse_args()
    base, key = a.base_url.rstrip("/"), a.api_key
    run = f"contract-{int(time.time())}"

    code, _, _ = http("GET", f"{base}/v1/health")
    check("GET /v1/health -> 200", code == 200, f"HTTP {code}")
    code, _, meta = http("GET", f"{base}/v1/meta", key)
    flat = {**(meta or {}), **((meta or {}).get("system") or {})}
    need = ["llm_model", "embedding_model", "index_version", "lex_snapshot_date"]
    check("GET /v1/meta has model/embedding/index/snapshot", code == 200 and all(flat.get(k) for k in need),
          f"HTTP {code}, missing={[k for k in need if not flat.get(k)]}")

    code, _, _ = http("POST", f"{base}/v1/rag/batch", "wrong-key", body(run, f"{run}-auth", QUESTIONS[:1]))
    check("wrong API key -> 401/403", code in (401, 403), f"HTTP {code}")

    eleven = [{"id": f"x{i}", "question": "Uy-joy mulkdorining majburiyatlari qanday?"} for i in range(11)]
    code, _, _ = http("POST", f"{base}/v1/rag/batch", key, body(run, f"{run}-11", eleven))
    check("11 items -> 400", code == 400, f"HTTP {code}")
    dup = [QUESTIONS[0], dict(QUESTIONS[1], id="c1")]
    code, _, _ = http("POST", f"{base}/v1/rag/batch", key, body(run, f"{run}-dup", dup))
    check("duplicate item id -> 400", code == 400, f"HTTP {code}")
    code, _, _ = http("POST", f"{base}/v1/rag/batch", key, body(run, f"{run}-empty", [{"id": "e1", "question": ""}]))
    check("empty question -> 400", code == 400, f"HTTP {code}")

    b1 = body(run, f"{run}-main", QUESTIONS)
    code, r1, el = post(base, key, b1)
    check("valid batch -> 200 (or 202 + GET)", code == 200, f"HTTP {code}")
    check("sync answer <= 120 s (else must be async 202)", el <= 120, f"{el:.1f}s")
    if code != 200:
        print("cannot continue without a valid response")
        return
    probs = validate_batch_response(r1, [q["id"] for q in QUESTIONS])
    check("response contract (§4: ids, citations, contexts, chunk_id ⊂ contexts, system)", not probs,
          "; ".join(probs[:8]))
    by_id = {r["id"]: r for r in r1.get("results", [])}
    u = by_id.get(UNANSWERABLE, {})
    check("unanswerable question -> no_answer: true", bool(u.get("no_answer")) or u.get("status") == "no_answer",
          f"answer={str(u.get('answer'))[:80]}")
    text = " ".join(str(r.get("answer") or "") for r in r1.get("results", []))
    check("no mojibake / broken o‘ g‘ in answers", not any(m in text for m in MOJIBAKE))
    for qid, lang in (("c3", "uz-Cyrl"), ("c4", "ru")):
        r = by_id.get(qid, {})
        check(f"{lang} question answered (status ok)", r.get("status") == "ok" and not r.get("no_answer"),
              f"status={r.get('status')} no_answer={r.get('no_answer')}")

    code, r2, _ = post(base, key, b1)
    check("same batch_id resent -> stored result (idempotent)", code == 200 and r2.get("results") == r1.get("results"))
    code, _, r3 = http("GET", f"{base}/v1/rag/batch/{b1['batch_id']}", key)
    check("GET /v1/rag/batch/{id} returns completed result",
          code == 200 and r3.get("status") == "completed" and r3.get("results") == r1.get("results"), f"HTTP {code}")
    changed = copy.deepcopy(b1)
    changed["items"][0]["question"] += " (o'zgartirilgan)"
    code, _, _ = http("POST", f"{base}/v1/rag/batch", key, changed)
    check("same batch_id, different content -> 409", code == 409, f"HTTP {code}")

    code, r4, _ = post(base, key, body(run, f"{run}-repeat", QUESTIONS))
    check("temperature=0 + seed: same questions -> same sources", code == 200 and sig(r4) == sig(r1))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
