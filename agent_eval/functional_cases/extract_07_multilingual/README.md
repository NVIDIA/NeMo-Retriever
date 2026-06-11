# Functional test suite — EXTRACT: standalone multi-lingual doc extraction

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI (`retriever ingest` / `retriever query`). Each test is a self-contained triple — a
prompt, a per-case `cases/<id>/data/` folder, and an expected output naming the correct
`retriever` subcommand(s) and the right **OCR v2 language routing**.

---

## The user task under test

> **JTBD: EXTRACT — P0.** "Standalone multi-lingual doc extraction — including non-Latin
> (CJK and similar) and Latin scripts." The skill must pick the correct `nemotron-ocr-v2`
> language mode per document and, for languages **outside** the OCR v2 trained set, relay to
> the user that the language is **not supported** rather than silently emit garbage.

**Success criteria for the row (operational pass, binary):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | (1) ingest each doc with the right `--ocr-version v2 --ocr-lang <selector>`, non-zero chunks; (2) extracted text preserves the SOURCE script (no romanization / no `?` placeholders); (3) same-language round-trip query returns the indexed row in top-5; (4) untrained language → the skill **gates** (declines, or invokes + sanity-checks + flags `language_unsupported`) instead of indexing garbage |
| Time | **medium** — each ingest+query ≤ 2 min |
| Trigger rate | ≥ 95% — a "pull the text out of this `<language>` doc" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest … --ocr-version v2 --ocr-lang <english\|multi>` then `retriever query`; **no manual OCR-model override** |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased):
- *"Pull the text out of this Japanese form."*
- *"Read this Chinese form and tell me what it says."*
- *"Extract the details from this Russian invoice — what's the total?"*
- *"Make this Korean document searchable."*

---

## How the CLI handles language (grounded, not guessed)

Verified against `retriever ingest --help` and `retriever ingest <file> --dry-run` (offline):

- **`--ocr-version`** accepts `[v1|v2]`; the resolved-plan **default is `v2`**.
- **`--ocr-lang`** accepts **only `[multi|english]`** — *not* the freeform `en`/`fr`/`zh`/`ar`
  selectors the spec sketches. The resolved-plan default is `ocr_lang: null`, which behaves as
  the **multilingual** mode. So:
  - the spec's **"en" flag** → `--ocr-lang english`;
  - the spec's **"multi (or default)"** → `--ocr-lang multi` *or* omit the flag.
- **OCR v2 trained languages (SCOPE):** English, Chinese (Simplified + Traditional),
  Japanese, Korean, Russian. Anything else (e.g. **Dutch**, French, Arabic) is **out of scope**
  → the negative-test **gate** (category **f**).
- **Language metadata** is separate from routing: each row carries
  `text_metadata.language` from **langdetect**. It is metadata-only (`UNKNOWN` acceptable) and
  is *not* used to choose the OCR mode — but it is the natural signal for the option-(b) gate.
- **Embedder** `nvidia/llama-nemotron-embed-1b-v2` must tolerate non-Latin scripts (Cyrillic,
  CJK) for the same-language round-trip query to return the indexed row.

**Language-routing decision the skill must make:**

| Document language | Trained? | Correct route |
|---|---|---|
| English | yes | `--ocr-lang english` |
| Russian / Chinese / Japanese / Korean | yes | `--ocr-lang multi` or default |
| Mixed (e.g. en+ru) | yes | `--ocr-lang multi` or default |
| Dutch / French / Arabic / … | **no** | **GATE** — relay "not supported" (don't silently index garbage) |

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `ml-001` | **Baseline.** English doc → `--ocr-lang english`, non-zero chunks, loop closes. | `ingest`, `query` |
| 2 | `ml-002` | **First non-Latin script.** Russian (trained) → multi/default; Cyrillic **preserved** (not romanized, no `?`). | `ingest`, `query` |
| 3 | `ml-003` | **Two scripts in one doc.** Russian+English mixed → one `multi` pass recovers a Cyrillic **and** a Latin token. | `ingest`, `query` |
| 4 | `ml-004` | **End-to-end round-trip.** A Russian-language query embeds and returns the indexed Russian row in top-5 (embedder tolerates Cyrillic). | `ingest`, `query` |
| 5 | `ml-005` | **Acceptance gate.** Dutch (untrained) → skill **gates** (declines or flags `language_unsupported`); does NOT silently index garbage. | gate / `ingest`+sanity-check |

The ladder: T1 proves the English route works; T2 adds the first non-Latin script and the
**script-preservation** check; T3 puts two scripts in a single document; T4 proves the text
**round-trips** through the embedder (extract → embed → index → retrieve) in the source script;
T5 is the composed acceptance gate — the correct **routing decision** across the
supported/unsupported boundary, where "do nothing / decline" is the right answer.

---

### T1 — `ml-001` · English baseline  *(complexity 1)*
- **Satisfies:** the EXTRACT row at its simplest — the English-mode route.
- **Data:** `cases/ml-001/data/test.pdf` (1-page placeholder text + an image reference).
- **Expected:** `$RETRIEVER ingest data/test.pdf --ocr-version v2 --ocr-lang english` →
  `$RETRIEVER query "What does the document say?" --top-k 5`. Non-zero rows; cite `test.pdf` p1.

### T2 — `ml-002` · Russian invoice (trained, non-Latin)  *(complexity 2)*
- **Satisfies:** the non-Latin clause + script preservation.
- **Data:** `cases/ml-002/data/invoice_001.jpg` (Russian invoice; total **20 400 000** RUB,
  seller **ООО «Аврора Технологии»** — from `multilingual/ground-truth.json`).
- **Expected:** ingest with `--ocr-lang multi` (or default), **not** `english`; extracted text
  contains Cyrillic (`СЧЁТ-ФАКТУРА`, `Аврора`); query returns the total, citing `invoice_001.jpg` p1.

### T3 — `ml-003` · Russian+English mixed contract  *(complexity 3)*
- **Satisfies:** the mixed-script clause (spec's en+ar / en+fr, realized with en+ru).
- **Data:** `cases/ml-003/data/contract_001.jpg` (bilingual contract; value **1,250,000** USD,
  Russian party **OOO Восход**).
- **Expected:** one `--ocr-lang multi` pass recovers both a Latin token (`1,250,000`) and a
  Cyrillic token (`Восход`); both queries cite `contract_001.jpg` p1.

### T4 — `ml-004` · same-language round-trip  *(complexity 4)*
- **Satisfies:** the embedder-tolerates-the-script + round-trip retrieval clause.
- **Data:** `cases/ml-004/data/invoice_001.jpg` (the Russian invoice again).
- **Expected:** ingest (multi) with `--embed-model-name nvidia/llama-nemotron-embed-1b-v2` →
  a **Russian** query (`Кто продавец?`) returns the invoice row in top-5; answer surfaces the
  seller `Аврора`, citing `invoice_001.jpg` p1.

### T5 — `ml-005` · acceptance gate: untrained language (Dutch)  *(complexity 5)*
- **Satisfies:** negative-test category **(f)** — the routing decision across the
  supported/unsupported boundary.
- **Data:** `cases/ml-005/data/factuur_001.jpg` (Dutch invoice; ground-truth total **90.000**
  EUR — but Dutch is **not** in the OCR v2 trained set, and `supported_by_ocr_v2: false` in the
  ground-truth file).
- **Expected:** the skill **gates** — either (a) declines before invoking and relays "Dutch is
  not supported by nemotron-ocr-v2 (supported: en/zh/ja/ko/ru)", or (b) invokes `--ocr-lang
  multi`, then sanity-checks `detect_language()` (→ `nl`) and flags `language_unsupported`. It
  must **not** present the English-biased mangled text as a confident answer.
- **Caveat (built into the test):** Dutch is Latin-script, so OCR v2's English variant *will*
  emit plausible-looking bytes — the failure mode is exactly **silent garbage that looks
  extracted**. The pass is the routing decision, not a non-zero byte count.

---

## Running / grading

Mount each case's `data/` into the agent workdir, give it the `prompt`, and grade against
`pass_when` in `cases.json`. The checks unique to this suite: **(a)** the correct `--ocr-lang`
per document language (`english` vs `multi`/default); **(b)** the extracted text preserves the
source script (no romanization / no `?` placeholders); **(c)** a same-language query round-trips;
**(d)** for an untrained language, the skill gates instead of silently indexing garbage.

**Note on live runs — not run live.** Expected outputs are grounded in the CLI source +
`--help` + an offline `--dry-run` (which confirmed `ocr_version=v2`, `ocr_lang=null` as defaults)
and in `multilingual/ground-truth.json` — they have **not** been executed live. A live run hits
the **hosted** `ai.api.nvidia.com` OCR v2 endpoint + the hosted embedder (billable, needs
`NVIDIA_API_KEY`); it would capture real row counts, the actual extracted strings (to verify
Cyrillic preservation and the Dutch mangling), top-5 ranks for the round-trip, langdetect values,
and per-doc latencies/token baselines.

**Fixture-language-gap note.** The spec's ideal language set is **en / fr / zh / ar / en-fr /
en-ar**. Only a subset is on disk: **en** (data PDFs), **ru**, **ru+en mixed**, and **nl**
(`/raid/retriever-sdg-v3/test-data/multilingual/`). This suite therefore exercises the routing
**logic** across the supported/unsupported boundary using **ru** (trained, non-Latin proxy for
zh/ja/ko) and **nl** (untrained Latin proxy for the gate), with **en+ru** standing in for the
mixed case. The genuinely missing legs — **CJK (zh/ja/ko)** trained extraction and **Arabic
(RTL) / French** — have **no fixture** and should be added as `ml-006…` once those documents
land. CJK in particular is the highest-value gap because it is the canonical "non-Latin" target
named in the user task and the seed queries (Japanese/Chinese/Korean forms).
