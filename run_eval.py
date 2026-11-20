#!/usr/bin/env python3
"""Send the test set to POST /v1/rag/batch, batch by batch (b01 -> b10), and save
every response to runs/<run_id>/bNN.json.  Spec §1 steps 2-4, §3, §5.

  export RAG_BASE_URL=https://...   RAG_API_KEY=...
  python run_eval.py --testset testset_uyjoy.csv --run-id eval-2026-11-08
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import date

from common import http, validate_batch_response

BACKOFF = [5, 15, 45]  # spec §5: 500/503/504 -> retry up to 3 times


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_batches(path):
    """Only id / question / language go to the API. Expected answers stay here."""
    batches = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = {"id": row["id"], "question": row["question"]}
            if row.get("language"):
                item["language"] = row["language"]
            batches.setdefault(row["batch"], []).append(item)
    return OrderedDict(sorted(batches.items()))


def poll(base, key, batch_id, interval=5, max_wait=1800):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(interval)
        code, _, data = http("GET", f"{base}/v1/rag/batch/{batch_id}", key)
        st = data.get("status") if isinstance(data, dict) else None
        if code == 200 and st in ("completed", "failed"):
            return data
        log(f"  poll {batch_id}: HTTP {code}, status={st}")
    raise RuntimeError(f"{batch_id}: still not completed after {max_wait}s")


def send_batch(base, key, body):
    attempt = 0
    while True:
        try:
            code, headers, data = http("POST", f"{base}/v1/rag/batch", key, body)
        except Exception as e:  # network error / client timeout
            code, headers, data = None, {}, {"_exception": str(e)}
        if code == 200:
            return data
        if code == 202:
            log(f"  202 accepted (async), polling {body['batch_id']}")
            return poll(base, key, body["batch_id"])
        if code == 429:
            wait = int(float(headers.get("retry-after", 5)))
            log(f"  429 rate limit, waiting {wait}s")
            time.sleep(wait)
            continue
        if code in (None, 500, 502, 503, 504) and attempt < len(BACKOFF):
            log(f"  {code or 'network error'} -> retry in {BACKOFF[attempt]}s")
            time.sleep(BACKOFF[attempt])
            attempt += 1
            continue
        hint = {400: "fix the request", 401: "check API key", 403: "check API key",
                409: "batch_id reused with different content -> use a new run_id"}.get(code, "")
        raise RuntimeError(f"{body['batch_id']}: HTTP {code} {hint}\n"
                           f"{json.dumps(data, ensure_ascii=False)[:500]}")


def save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("RAG_BASE_URL", "http://localhost:8080"))
    ap.add_argument("--api-key", default=os.environ.get("RAG_API_KEY"))
    ap.add_argument("--testset", default="testset_uyjoy.csv")
    ap.add_argument("--run-id", default=f"eval-{date.today().isoformat()}")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--as-of-date", default=date.today().isoformat())
    ap.add_argument("--resume", action="store_true", help="skip batches already saved")
    ap.add_argument("--retry-item-errors", type=int, default=1,
                    help="resend batch (same batch_id) if an item has retryable error")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")

    out_dir = os.path.join(a.out, a.run_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        code, _, meta = http("GET", f"{base}/v1/meta", a.api_key)
    except Exception as e:
        raise RuntimeError(f"cannot reach {base}: {e}")
    save(os.path.join(out_dir, "meta.json"), {"http": code, "meta": meta})
    log(f"meta: HTTP {code} {json.dumps(meta, ensure_ascii=False)[:200]}")

    batches = load_batches(a.testset)
    total = len(batches)
    all_problems = {}
    for idx, (b, items) in enumerate(batches.items(), 1):
        path = os.path.join(out_dir, f"{b}.json")
        if a.resume and os.path.exists(path):
            log(f"{b}: already saved, skipping")
            continue
        body = {
            "run_id": a.run_id,
            "batch_id": f"{a.run_id}-{b}",
            "batch_index": idx,
            "total_batches": total,
            "options": {"top_k": a.top_k, "return_contexts": True, "return_debug": True,
                        "temperature": 0, "seed": a.seed, "as_of_date": a.as_of_date,
                        "answer_language": "auto"},
            "items": items,
        }
        save(os.path.join(out_dir, f"{b}.request.json"), body)
        log(f"{b}: sending {len(items)} questions ({body['batch_id']})")
        t0 = time.time()
        resp = send_batch(base, a.api_key, body)

        # spec §3: retryable item error -> resend with the SAME batch_id
        for _ in range(a.retry_item_errors):
            retry = [r["id"] for r in resp.get("results", [])
                     if r.get("status") in ("error", "timeout") and (r.get("error") or {}).get("retryable")]
            if not retry:
                break
            log(f"  retryable errors in {retry}, resending same batch_id")
            resp2 = send_batch(base, a.api_key, body)
            if resp2 == resp:
                log("  server returned the stored result (idempotency) - errors stay; see README §Open questions")
                break
            resp = resp2

        save(path, resp)
        problems = validate_batch_response(resp, [i["id"] for i in items])
        if problems:
            all_problems[b] = problems
            save(os.path.join(out_dir, f"{b}.problems.json"), problems)
        s = resp.get("summary", {})
        log(f"{b}: done in {time.time() - t0:.1f}s  summary={s}  contract_problems={len(problems)}")

    log(f"run saved to {out_dir}/")
    if all_problems:
        log(f"WARNING: contract problems in {list(all_problems)} (see *.problems.json)")
    log(f"next: python evaluate.py {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        log(f"FATAL: {e}")
        sys.exit(1)
