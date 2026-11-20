#!/usr/bin/env python3
"""MOCK of the RAG API (spec §2-5) for dry-running the harness. Texts are SYNTHETIC,
not real law. Run:  MOCK_API_KEY=test-key python mock_server.py --port 8080
"""
import argparse
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import re as _re, unicodedata as _ud
_APOS = "\u02bb\u02bc\u2018\u2019`\u00b4"


def norm_text(s):
    s = _ud.normalize("NFC", s or "").lower()
    for ch in _APOS:
        s = s.replace(ch, "'")
    return _re.sub(r"\s+", " ", s).strip()


API_KEY = os.environ.get("MOCK_API_KEY", "test-key")
TZ = timezone(timedelta(hours=5))
SYSTEM = {"pipeline_version": "mock-0.1", "llm_model": "mock-llm", "embedding_model": "mock-embed",
          "reranker": None, "index_version": "mock-idx-uyjoy", "lex_snapshot_date": "2026-11-01"}
TITLES = {"UJK": "Uy-joy kodeksi", "SHIRKAT": "Shirkatlar to'g'risida qonun", "KKUB": "Ko'p kvartirali uylarni boshqarish",
          "KOMMUNAL_QOIDA": "Kommunal xizmatlar qoidalari", "FK": "Fuqarolik kodeksi", "MJTK": "MJtK",
          "SUV": "Ichimlik suvi", "ELEKTR": "Elektr energetikasi"}
# (doc key, article, synthetic text) — keyword-rich placeholders only
CHUNKS = [
    ("UJK", "1-modda", "[MOCK] uy-joy fondi turlari: xususiy, davlat va jamoat uy-joy fondi."),
    ("UJK", "2-modda", "[MOCK] ko'p kvartirali uyda umumiy mulk: zinapoya, tom, yerto'la, chordoq, lift, muhandislik tarmoqlari. "
                       "общее имущество многоквартирного дома. кўп квартирали уйда умумий мулк."),
    ("UJK", "3-modda", "[MOCK] turar joyni noturar joyga o'tkazish va qayta rejalashtirish (pereplanirovka) hokimlik ruxsati bilan. "
                       "перепланировка квартиры согласование."),
    ("UJK", "4-modda", "[MOCK] uy-joy mulkdori huquqlari va majburiyatlari, oila a'zolari foydalanishi, xususiylashtirish."),
    ("SHIRKAT", "5-modda", "[MOCK] shirkat tashkil etish, umumiy yig'ilish, rais saylash, taftish komissiyasi, a'zolikdan chiqish. "
                           "ширкат умумий йиғилиши чақирилади. товарищество собственников жилья права члена."),
    ("SHIRKAT", "6-modda", "[MOCK] umumiy mulkni saqlash xarajatlari, kapital ta'mirlash, tom ta'miri, jamg'arma ajratmalari, obodonlashtirish. "
                           "капитальный ремонт крыши расходы на содержание."),
    ("KKUB", "7-modda", "[MOCK] ko'p kvartirali uyni boshqarish usullari, boshqaruvchi tashkilotni tanlash, shartnoma."),
    ("KOMMUNAL_QOIDA", "8-modda", "[MOCK] kommunal xizmatlar iste'molchisi huquqlari, to'lov, hisoblagich, xizmatni to'xtatish, penya, sifatsiz xizmat. "
                                  "коммунал хизматлар тўлов. отключить воду за долги уведомление."),
    ("FK", "9-modda", "[MOCK] turar joy ijarasi shartnomasi yozma shaklda, subijara, bekor qilish, ijarachi, mulk huquqi, oldi-sotdi, meros, zarar undirish. "
                      "турар жойни ижарага бериш шартномаси."),
    ("MJTK", "10-modda", "[MOCK] jarima BHM baravari: kommunal to'lov, ruxsatsiz qayta rejalashtirish, suv, elektr, gaz noqonuniy foydalanish."),
    ("SUV", "11-modda", "[MOCK] ichimlik suvi sifati, suv ta'minoti uzilishi. ичимлик суви таъминоти мурожаат."),
    ("ELEKTR", "12-modda", "[MOCK] elektr energiyasi, kuchlanish, maishiy texnika zarari, svet, tarif."),
]
TH = 0.2
STORE, LOCK = {}, threading.Lock()


def stems(s):
    return {t[:5] for t in re.findall(r"\w+", norm_text(s)) if len(t) >= 4}


def chunk_obj(i):
    k, art, text = CHUNKS[i]
    doc = f"lex:MOCK-{k}"
    return k, art, text, doc, f"{doc}#art-{art.split('-')[0]}#p1"


def answer_item(item, top_k):
    q = stems(item["question"])
    scored = sorted(((len(q & stems(c[2])) / max(len(q), 1), i) for i, c in enumerate(CHUNKS)), key=lambda x: (-x[0], x[1]))
    h = int(hashlib.md5(item["question"].encode()).hexdigest(), 16)
    timing = {"retrieval_ms": 200 + h % 200, "rerank_ms": 100, "generation_ms": 1500 + h % 3000}
    timing["total_ms"] = sum(timing.values())
    ctxs = []
    for rank, (s, i) in enumerate(scored[:top_k], 1):
        k, art, text, doc, cid = chunk_obj(i)
        ctxs.append({"rank": rank, "chunk_id": cid, "doc_id": doc, "article": art, "text": text,
                     "retrieval_score": round(s, 3), "rerank_score": round(min(1.0, s * 1.3), 3),
                     "used_in_answer": rank == 1 and s >= TH})
    base = {"id": item["id"], "answer_language": item.get("language") or "uz-Latn", "contexts": ctxs,
            "debug": {"rewritten_query": item["question"], "retrieved_total": len(CHUNKS)},
            "timing": timing, "usage": {"input_tokens": 900, "output_tokens": 80}, "error": None}
    if not scored or scored[0][0] < TH:
        return {**base, "status": "no_answer", "answer": "Qonunchilikda bu savolga javob topilmadi.",
                "no_answer": True, "confidence": 0.1, "citations": []}
    k, art, text, doc, cid = chunk_obj(scored[0][1])
    cit = {"ref": 1, "doc_id": doc, "doc_title": TITLES.get(k, k), "doc_type": "kodeks",
           "url": f"https://lex.uz/docs/MOCK-{k}", "article": art, "part": "1-qism", "edition_date": "2026-01-01",
           "doc_status": "amalda", "chunk_id": cid, "quote": text[:70]}
    return {**base, "status": "ok", "answer": f"{text} [1].", "no_answer": False,
            "confidence": round(scored[0][0], 2), "citations": [cit]}


def validate(b):
    items = b.get("items")
    if not b.get("run_id") or not b.get("batch_id"):
        return "run_id and batch_id required"
    if not isinstance(items, list) or not 1 <= len(items) <= 10:
        return "items must contain 1-10 questions"
    ids = [i.get("id") for i in items]
    if len(set(ids)) != len(ids) or not all(ids):
        return "item ids must be unique and non-empty"
    if any(not (i.get("question") or "").strip() or len(i["question"]) > 2000 for i in items):
        return "question must be 1-2000 chars"
    return None


class H(BaseHTTPRequestHandler):
    def send(self, code, obj, headers=None):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authed(self):
        if self.headers.get("Authorization") != f"Bearer {API_KEY}":
            self.send(401, {"error": {"code": "UNAUTHORIZED", "message": "bad API key"}})
            return False
        return True

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
                    "<title>RAG Uy-joy API</title><body style='font:16px system-ui;max-width:640px;margin:40px auto;padding:0 16px'>"
                    "<h1>RAG Uy-joy API is running &#9989;</h1>"
                    "<p>This is the demo/mock API (synthetic answers, not real law text).</p>"
                    "<ul><li><a href='/v1/health'>/v1/health</a> &rarr; status</li></ul>"
                    "<p>Try it: <code>curl -X POST URL/v1/rag/query -H 'Authorization: Bearer test-key' "
                    "-H 'Content-Type: application/json' -d '{\"question\":\"Umumiy mulkka nimalar kiradi?\"}'</code></p></body>").encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page))); self.end_headers(); self.wfile.write(page); return
        if self.path == "/v1/health":
            return self.send(200, {"status": "ok"})
        if not self.authed():
            return
        if self.path == "/v1/meta":
            return self.send(200, SYSTEM)
        m = re.fullmatch(r"/v1/rag/batch/(.+)", self.path)
        if m and m.group(1) in STORE:
            return self.send(200, STORE[m.group(1)][1])
        self.send(404, {"error": {"code": "NOT_FOUND"}})

    def do_POST(self):
        if not self.authed():
            return
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
        except Exception:
            return self.send(400, {"error": {"code": "BAD_JSON"}})
        if self.path == "/v1/rag/query":
            return self.send(200, answer_item({"id": "single", **b}, (b.get("options") or {}).get("top_k", 5)))
        if self.path != "/v1/rag/batch":
            return self.send(404, {"error": {"code": "NOT_FOUND"}})
        err = validate(b)
        if err:
            return self.send(400, {"error": {"code": "VALIDATION_ERROR", "message": err}})
        h = hashlib.sha256(json.dumps(b, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with LOCK:
            if b["batch_id"] in STORE:
                if STORE[b["batch_id"]][0] != h:
                    return self.send(409, {"error": {"code": "BATCH_ID_CONFLICT"}})
                return self.send(200, STORE[b["batch_id"]][1])
        now = datetime.now(TZ)
        res = [answer_item(i, (b.get("options") or {}).get("top_k", 5)) for i in b["items"]]
        resp = {"run_id": b["run_id"], "batch_id": b["batch_id"], "status": "completed", "system": SYSTEM,
                "timing": {"received_at": now.isoformat(timespec="seconds"),
                           "completed_at": now.isoformat(timespec="seconds"),
                           "total_ms": sum(r["timing"]["total_ms"] for r in res)},
                "summary": {"total": len(res), "ok": sum(r["status"] == "ok" for r in res),
                            "no_answer": sum(r["status"] == "no_answer" for r in res), "error": 0},
                "results": res}
        with LOCK:
            STORE[b["batch_id"]] = (h, resp)
        self.send(200, resp)

    def log_message(self, *a):
        pass



# ---- WSGI entrypoint (Vercel looks for `app`) ----
PAGE = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>RAG Uy-joy API</title><body style='font:16px system-ui;max-width:640px;margin:40px auto;padding:0 16px'>"
        "<h1>RAG Uy-joy API is running &#9989;</h1><p>This is the demo/mock API (synthetic answers, not real law text).</p>"
        "<ul><li><a href='/v1/health'>/v1/health</a> &rarr; status</li></ul></body>").encode()


def app(environ, start_response):
    method, path = environ["REQUEST_METHOD"], environ.get("PATH_INFO", "/")
    if not path.startswith("/v1/"):  # Vercel may rewrite "/" to "/api/index"
        path = "/"

    def out(code, obj, ctype="application/json; charset=utf-8"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        start_response(code, [("Content-Type", ctype), ("Content-Length", str(len(data)))])
        return [data]

    if method == "GET" and path in ("/", "/index.html"):
        return out("200 OK", PAGE, "text/html; charset=utf-8")
    if method == "GET" and path == "/v1/health":
        return out("200 OK", {"status": "ok"})
    if environ.get("HTTP_AUTHORIZATION") != f"Bearer {API_KEY}":
        return out("401 Unauthorized", {"error": {"code": "UNAUTHORIZED", "message": "bad API key"}})
    if method == "GET" and path == "/v1/meta":
        return out("200 OK", SYSTEM)
    m = re.fullmatch(r"/v1/rag/batch/(.+)", path)
    if method == "GET" and m and m.group(1) in STORE:
        return out("200 OK", STORE[m.group(1)][1])
    if method == "POST":
        try:
            n = int(environ.get("CONTENT_LENGTH") or 0)
            b = json.loads(environ["wsgi.input"].read(n).decode("utf-8"))
        except Exception:
            return out("400 Bad Request", {"error": {"code": "BAD_JSON"}})
        k = (b.get("options") or {}).get("top_k", 5)
        if path == "/v1/rag/query":
            return out("200 OK", answer_item({"id": "single", **b}, k))
        if path == "/v1/rag/batch":
            err = validate(b)
            if err:
                return out("400 Bad Request", {"error": {"code": "VALIDATION_ERROR", "message": err}})
            h = hashlib.sha256(json.dumps(b, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if b["batch_id"] in STORE:
                if STORE[b["batch_id"]][0] != h:
                    return out("409 Conflict", {"error": {"code": "BATCH_ID_CONFLICT"}})
                return out("200 OK", STORE[b["batch_id"]][1])
            now = datetime.now(TZ)
            res = [answer_item(i, k) for i in b["items"]]
            resp = {"run_id": b["run_id"], "batch_id": b["batch_id"], "status": "completed", "system": SYSTEM,
                    "timing": {"received_at": now.isoformat(timespec="seconds"),
                               "completed_at": now.isoformat(timespec="seconds"),
                               "total_ms": sum(r["timing"]["total_ms"] for r in res)},
                    "summary": {"total": len(res), "ok": sum(r["status"] == "ok" for r in res),
                                "no_answer": sum(r["status"] == "no_answer" for r in res), "error": 0},
                    "results": res}
            STORE[b["batch_id"]] = (h, resp)
            return out("200 OK", resp)
    return out("404 Not Found", {"error": {"code": "NOT_FOUND"}})



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    port = ap.parse_args().port
    print(f"mock RAG API on http://localhost:{port}  (key={API_KEY})")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
