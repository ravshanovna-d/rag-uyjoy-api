"""Shared helpers for the lex.uz RAG test kit (stdlib only)."""
import json
import re
import unicodedata
import urllib.error
import urllib.request

# ---------- text helpers ----------
_APOS = "ʻʼ‘’`´"


def norm_text(s):
    """NFC, lowercase, unify o' / g' apostrophes, collapse whitespace."""
    s = unicodedata.normalize("NFC", s or "").lower()
    for ch in _APOS:
        s = s.replace(ch, "'")
    return re.sub(r"\s+", " ", s).strip()


def article_num(a):
    """'123-modda' / 'статья 123' / '123' -> '123'."""
    m = re.search(r"\d+", str(a or ""))
    return m.group(0) if m else None


# ---------- HTTP ----------
def _parse(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {"_raw": raw.decode("utf-8", "replace")}


def http(method, url, key=None, body=None, timeout=180):
    """Returns (status, headers_lowercase, json). Raises on network errors."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json; charset=utf-8")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, _parse(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, _parse(e.read())


# ---------- response contract (spec §4) ----------
REQ_TOP = ["run_id", "batch_id", "status", "system", "timing", "results"]
REQ_SYSTEM = ["pipeline_version", "llm_model", "embedding_model", "index_version", "lex_snapshot_date"]
REQ_RESULT = ["id", "status", "answer", "no_answer", "citations", "contexts"]
REQ_CIT = ["ref", "doc_id", "url", "article", "part", "edition_date", "doc_status", "chunk_id", "quote"]
REQ_CTX = ["rank", "chunk_id", "doc_id", "article", "text", "retrieval_score", "rerank_score"]
STATUSES = {"ok", "no_answer", "error", "timeout"}


def validate_batch_response(resp, sent_ids):
    """Return a list of contract violations (empty list = OK)."""
    p = []
    if not isinstance(resp, dict):
        return ["response is not a JSON object"]
    for k in REQ_TOP:
        if k not in resp:
            p.append(f"missing top-level '{k}'")
    for k in REQ_SYSTEM:
        if not (resp.get("system") or {}).get(k):
            p.append(f"missing system.{k}")
    results = resp.get("results") or []
    ids = [r.get("id") for r in results]
    if len(ids) != len(set(ids)):
        p.append("duplicate ids in results")
    if sorted(map(str, ids)) != sorted(map(str, sent_ids)):
        p.append(f"result ids != item ids (missing={set(sent_ids) - set(ids)}, extra={set(ids) - set(sent_ids)})")
    for r in results:
        rid = r.get("id")
        for k in REQ_RESULT:
            if k not in r:
                p.append(f"{rid}: missing '{k}'")
        st = r.get("status")
        if st not in STATUSES:
            p.append(f"{rid}: bad status '{st}'")
        if st in ("error", "timeout"):
            if not isinstance(r.get("error"), dict) or "code" not in r["error"]:
                p.append(f"{rid}: status={st} but no error object")
            continue
        if (r.get("timing") or {}).get("total_ms") is None:
            p.append(f"{rid}: missing timing.total_ms")
        ctx = r.get("contexts") or []
        ctx_ids = {c.get("chunk_id") for c in ctx}
        for c in ctx:
            for k in REQ_CTX:
                if k not in c:
                    p.append(f"{rid}: context rank {c.get('rank')} missing '{k}'")
        cits = r.get("citations") or []
        if st == "ok" and not r.get("no_answer") and not cits:
            p.append(f"{rid}: answered but has no citations")
        refs = set()
        for c in cits:
            refs.add(c.get("ref"))
            for k in REQ_CIT:
                if k not in c:
                    p.append(f"{rid}: citation ref {c.get('ref')} missing '{k}'")
            if c.get("chunk_id") not in ctx_ids:
                p.append(f"{rid}: citation chunk_id {c.get('chunk_id')} not in contexts")
        for n in re.findall(r"\[(\d+)\]", r.get("answer") or ""):
            if int(n) not in refs:
                p.append(f"{rid}: answer cites [{n}] but no citation with that ref")
    return p
