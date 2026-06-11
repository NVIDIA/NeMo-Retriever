# Synopsis — EXTRACT: standalone multi-lingual doc extraction

**What user task this covers.** A user hands the system documents in different languages —
including **non-Latin scripts** (Chinese, Japanese, Korean, Russian) and Latin scripts — and
wants the text pulled out and made searchable. Success means the skill drives the `retriever`
CLI to extract with the **correct OCR v2 language mode per document**, keeps the text in its
**original script** (no romanization, no `?` boxes), lets the user search it back in the same
language — and, crucially, when a document is in a language the OCR model was **not trained on**,
tells the user it's **not supported** instead of quietly producing garbage.

**How we test it.** Five agent prompts, each handing the agent one document and checking that it
(1) picks the right `--ocr-lang` (English documents → `english`; trained non-English and mixed
documents → `multi`/default), (2) returns non-zero extracted chunks, (3) preserves the source
script, (4) round-trips a same-language query, and (5) makes the right call at the
supported/unsupported boundary. We use the multilingual fixtures actually on disk — a Russian
invoice, a Russian+English contract, and a Dutch invoice — plus an English baseline PDF.

**The five tests, simplest to hardest:**

1. **English baseline** — extract an English PDF with the English OCR mode; confirm non-zero
   chunks and that the loop closes.
2. **Russian invoice** — first non-Latin script; route to the multilingual mode and confirm the
   Cyrillic text comes out as Cyrillic (not transliterated), recovering the invoice total.
3. **Russian+English contract** — two scripts in one document; a single multilingual pass must
   recover both a Cyrillic and a Latin fact.
4. **Same-language round-trip** — ask in Russian and get the indexed Russian document back in the
   top results, proving the embedder handles the non-Latin script end to end.
5. **Acceptance gate** — a Dutch invoice, whose language the OCR model was not trained on: the
   skill must decline (or extract, sanity-check, and flag "language not supported") rather than
   present mangled text as a confident answer.

**Why this order.** Each rung adds exactly one thing: first the simplest (English) route; then
the first non-Latin script and the script-preservation check; then two scripts in one document;
then proof that the text survives the full extract→embed→search round-trip in its own script;
finally the hardest call — recognizing an unsupported language and **gating** instead of emitting
garbage, which is the real acceptance criterion for this row.

**Coverage gap (stated up front).** The task names Japanese/Chinese/Korean and Arabic, but only
English, Russian, Russian+English, and Dutch fixtures exist on disk today. This suite proves the
routing **logic** across the supported/unsupported boundary using Russian (a trained non-Latin
language) and Dutch (an untrained Latin language). The missing **CJK** (zh/ja/ko) and **Arabic /
French** legs have no fixture yet and are flagged in the README to be added when those documents
become available.

**Status.** Tests are authored and grounded in the real CLI (`--help` + offline `--dry-run`) and
the multilingual ground-truth file; **not yet run live** (a live run hits the hosted OCR v2 and
embedding endpoints and needs an API key). See `README.md` for the full spec and `cases.json` for
the machine-gradable definitions.
