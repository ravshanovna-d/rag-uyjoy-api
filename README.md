# RAG API test kit — Uy-joy qonunchiligi va kommunal xo'jalik (lex.uz)

Implements the "RAG API — test spetsifikatsiyasi" (Nov 3, 2026), scoped to housing and
communal-services law instead of all of lex.uz. Python 3.9+, standard library only.

| File | Spec | What it does |
|---|---|---|
| `testset_uyjoy.csv` | §1.1–1.2, §6 | 100 questions, q001–q100, already split into b01–b10, every category mixed into every batch |
| `sources.json` | §6 | Map of document keys (UJK, SHIRKAT, FK …) to lex.uz IDs — **fill `lex_id` / `url`** |
| `run_eval.py` | §1.3–1.4, §3, §5 | Sends b01→b10 sequentially, handles 202/poll, 429 Retry-After, 5xx retries (5/15/45 s), saves `runs/<run_id>/bNN.json` and validates each response |
| `evaluate.py` | §1.5–1.6, §6 | All §6 metrics, `report.md`, `metrics.json`, manual `grading.csv`; `--compare` for two runs |
| `contract_check.py` | §7 | Developer checklist as 17 automated PASS/FAIL checks |
| `mock_server.py` | — | Spec-conformant fake API (synthetic text) to dry-run everything before the real API exists |

## Test set: 100 questions (spec §6 distribution)

| Category (`category`) | n | Example |
|---|---|---|
| `faktik` — bitta modda | 35 | Ko'p kvartirali uyda umumiy mulkka nimalar kiradi? |
| `raqamlar` — muddat, summa, foiz, BHM | 15 | Shirkat umumiy yig'ilishi necha foiz a'zo ishtirokida vakolatli? |
| `kop_modda` — bir nechta modda/hujjat | 15 | Ijarachi kommunal to'lamasa, mulkdor javobgarmi? |
| `ozgargan` — o'zgargan / kuchini yo'qotgan | 10 | Ijtimoiy normadan ortiq elektr uchun alohida tarif amaldami? |
| `til` — rus tili va kirill | 10 | 5 ru + 5 uz-Cyrl |
| `javobsiz` — qonundan tashqari | 10 | Toshkentda 2030-yilda kvartira narxi qancha bo'ladi? |
| `noaniq` — og'zaki | 5 | svet o'chib qoldi, muzlatkich buzildi, pulini qaytarishadimi |

Some questions deliberately use `ʻ` / `‘` apostrophes to test o' / g' handling.

### ⚠️ What you still must fill (ground truth)

I wrote questions and the expected **document** for each, but I did **not** invent
`expected_answer` or `expected_article`: legal ground truth must be copied from the current
lex.uz text, otherwise the benchmark measures against wrong answers. For each answerable row:

1. Open the document on lex.uz, find the article (and current edition).
2. Fill `expected_answer`, `expected_article` (e.g. `12-modda`), set `verified=true`.
3. Fill `lex_id` / `url` once per document in `sources.json` (or per row in `expected_doc_id`).
4. Multi-source rows: `|`-aligned, e.g. `expected_doc_key=FK|SHIRKAT`, `expected_article=9-modda|6-modda`.
5. Replace any question whose topic turns out not to be regulated (then it becomes a `javobsiz` case — or rewrite it).

Several entries in `sources.json` are marked *(aniqlansin)* — check the exact act name and
whether it is still in force. Metrics with no ground truth yet show `n/a`, not a fake number.

## Usage

```bash
export RAG_BASE_URL=https://<test-base-url>  RAG_API_KEY=<test-key>

python contract_check.py                           # §7: is the API ready for testing?
python run_eval.py --run-id eval-2026-11-08        # §1: 10 batches -> runs/eval-2026-11-08/
python evaluate.py runs/eval-2026-11-08            # §6: report.md, metrics.json, grading.csv
# grade grading.csv by hand (or LLM judge), then run evaluate.py again
python evaluate.py --compare runs/eval-2026-11-08 runs/eval-2026-11-20   # after system change
```

Dry run without the real API:
```bash
python mock_server.py --port 8080 &
RAG_BASE_URL=http://localhost:8080 RAG_API_KEY=test-key python contract_check.py
RAG_BASE_URL=http://localhost:8080 RAG_API_KEY=test-key python run_eval.py --run-id dry
python evaluate.py runs/dry --sources sources.mock.json
```

## How metrics are computed

- **Hit@5 (hujjat / modda), MRR** — over `contexts` sorted by `rank`; article match = same number and same `doc_id` (if known). MRR uses all returned contexts, 0 if not found.
- **Context precision** — automatic *proxy*: share of top-5 chunks from an expected document. True relevance needs judgment.
- **To'qilgan manba** — citation whose `chunk_id` is not in `contexts`, or `url` not on lex.uz.
- **Eskirgan norma** — citation with `doc_status` other than `amalda`.
- **Quote grounded** — normalized `quote` is a substring of its context text (automatic faithfulness proxy).
- **Answer correctness / faithfulness / citation accuracy / error cause** — from `grading.csv` (columns `correctness_0_1_2`, `faithful_0_1`, `citation_ok_0_1`, `error_cause` = retrieval / generatsiya / manba). `auto_cause=retrieval` is pre-filled when the right article was not in top-5. The sheet is created once and never overwritten.
- **Latency p50/p95** — per-question `timing.total_ms`; **error rate** counts error, timeout and missing results.

## Open questions found while implementing (add to spec §7 list)

1. **Retry vs idempotency conflict.** §3 says resend a batch with the *same* `batch_id` when an item has `retryable: true`; §2 says the same `batch_id` returns the *stored* result. Both cannot hold. Proposal: store results only for items that did not fail retryably, or let the tester resend just the failed items under `<batch_id>-r1`. The runner currently resends once and logs if the stored result came back.
2. `GET /v1/meta` field names aren't specified — the check accepts them top-level or under `system`.
3. Context precision and citation accuracy need a relevance judgment; agree whether an LLM judge is acceptable.
