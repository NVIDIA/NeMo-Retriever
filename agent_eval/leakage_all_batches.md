# Answer-location leakage — all 4 agent-eval batches

**Goal.** The four `*prompts.json` batches under `retriever-sdg-v3/runs/` are meant to
test whether an agent (Claude, Codex, …) benefits from the NRL retriever CLI versus
non-retriever solutions. A prompt that tells the agent **which file the answer is in**
defeats that test — the agent can open the file directly and the retriever shows no
benefit. This document records how many prompts leak the answer's location, under two
definitions, across all four batches.

**Batches analyzed**

| Batch | File | n |
|---|---|--:|
| batch_1 | `runs/vidore_v3_agent_evals_20260513_batch_1/agent_prompts.json` | 46 |
| batch_2 | `runs/vidore_v3_agent_evals_20260514_batch_2/agent_scenario_prompts.json` | 135 |
| batch_3 | `runs/nemo_retriever_agent_functional_batch_3/agent_functional_prompts.json` | 92 |
| batch_4 | `runs/agent_corpus_level_batch_4/agent_corpus_level_prompts.json` | 1210 |
| **Total** | | **1483** |

---

## Two definitions of "leak"

### 1. Filename/path leakage (weak — string-only)

Does the prompt text contain a **specific file path or filename** (not a glob)? This is
a pure string match on the prompt; it needs no ground-truth labels.

| Batch | n | leak: 1 file | leak: few files | glob/corpus | no path | **% leaking** |
|---|--:|--:|--:|--:|--:|--:|
| batch_1 | 46 | 19 | 9 | 17 | 1 | **60.9%** |
| batch_2 | 135 | 59 | 28 | 29 | 19 | **64.4%** |
| batch_3 | 92 | 31 | 3 | 0 | 58 | **37.0%** |
| batch_4 | 1210 | 0 | 0 | 1030 | 180 | **0.0%** |
| **Total** | 1483 | 109 | 40 | 1076 | 258 | **10.0%** |

This definition reports **0% for batch_4** because every batch_4 prompt uses a folder
glob (`test-data/.../pdfs/*.pdf`) and never names a literal file. That is correct for
*string* leakage but misses the real problem (see below).

### 2. Answer-location leakage (strong — entity/title)

Does the prompt **identify the gold answer document** — by filename, by naming the
source entity/title, by distinctive-token overlap, or by a curated alias/topic? This
matches each prompt against its gold doc(s) and is the definition used in
[`leakage_batch4.md`](leakage_batch4.md). It catches the case where a prompt says
"search all the PDFs" but names the company/report, which still pins the file.

This is the headline analysis; full results below.

---

## Method (strong definition)

For each prompt, join to its gold document(s) and flag a leak if **any** of these fire
(checked in order; each prompt is assigned to the first/strongest match, so the four
mechanisms are mutually exclusive):

| Mechanism | What it detects |
|---|---|
| **filename** | The gold doc's literal filename/path string appears in the prompt (no glob). |
| **entity-phrase** | The gold doc_id's distinctive tokens (years, alphanumeric codes like `KE0125029ENN`, and generic words like *report/annual/presentation* dropped) appear as a contiguous phrase, or a single distinctive proper-noun token appears. High precision. |
| **overlap** | ≥70% of the gold doc's distinctive tokens (and ≥2 of them) appear, not necessarily contiguous. Lower precision; spot-checked. |
| **alias** | A curated abbreviation/synonym/unique-topic maps 1:1 to the gold file but shares no literal tokens with its name (finance company nicknames, HR report topics, narrow pharma topics — same map as `leakage_batch4.md`). |

**Gold docs.** `relevant_pages[].doc_id` (batches 1, 2, 4, via the batch manifest) or
`source_refs` filenames (batch 3).

**Alias map** (carried over from the batch_4 analysis):
- *Finance (6 companies):* bank of america / bofa / merrill · citigroup / citi ·
  jpmorgan / jp morgan / jpm / chase · goldman · morgan stanley · wells fargo / wfc
- *HR (14 reports):* intra-EU labour mobility · public employment services ·
  future-oriented occupations · future of work · labour market transitions / skills
  investment · posting of / posted workers · joint employment report · domestic workers ·
  wage developments · undeclared (care) work · demographic perspective · working
  conditions / career development · employment and social developments
- *Pharma (unique-topic only):* drug/antimicrobial/antibiotic resistance →
  `drug_resistance_book` · dscsa → DSCSA webinar · medication error → DMEPA slides ·
  vaccine → `medicine_vaccine_book`

**Validation.** On batch_4 this reconstruction flags **281 token-only / 327
alias-augmented** vs. the authoritative **267 / 323** from `leakage_batch4.md`, with
313/323 exact agreement — within ~2%. The small gap is mostly bare-"CDER" tokens the
prior pass treated as a distinctive entity and this one treats as generic.

---

## Findings (strong definition)

| Batch | leaking | answerable | **% of answerable** | all prompts | % of all |
|---|--:|--:|--:|--:|--:|
| batch_1 — `agent_prompts.json` | 37 | 46 | **80.4%** | 46 | 80.4% |
| batch_2 — `agent_scenario_prompts.json` | 56 | 77 | **72.7%** | 135 | 41.5% |
| batch_3 — `agent_functional_prompts.json` | 65 | 84 | **77.4%** | 92 | 70.7% |
| batch_4 — `agent_corpus_level_prompts.json` | 327 | 1001 | **32.7%** | 1210 | 27.0% |
| **TOTAL** | **485** | **1208** | **40.1%** | **1483** | **32.7%** |

### Column definitions

- **leaking** — prompts whose text identifies the gold answer document (by any of the
  four mechanisms). The contaminated set.
- **answerable** — prompts that have a gold answer document at all (`relevant_pages` /
  `source_refs` non-empty). Leakage is only *possible* for these; it excludes
  refusal / ingest-only / extract-only / capability-gap tests that have no answer doc.
- **% of answerable** — `leaking ÷ answerable`. **The headline rate**: of prompts that
  have an answer to find, the fraction that reveal where it lives.
- **% of all** — `leaking ÷ total prompts`. Diluted by non-answerable prompts; differs
  from "% of answerable" only where a batch contains them (notably batch_2).

### Leak mechanism breakdown (how the file was revealed)

| Batch | filename | entity-phrase | overlap | alias |
|---|--:|--:|--:|--:|
| batch_1 | 28 | 7 | 1 | 1 |
| batch_2 | 47 | 6 | 0 | 3 |
| batch_3 | 55 | 10 | 0 | 0 |
| batch_4 | **0** | 180 | 8 | 139 |

---

## Why the two definitions disagree on batch_4 (0% vs 27%)

They measure different things, and they **agree** where they overlap — `leakage_batch4.md`
itself reports "Explicit `.pdf` filename matching a gold doc = 0 / 0.0%", identical to the
weak definition's 0%. The 27% comes entirely from **entity/topic-name leakage**: prompts
that use a `*.pdf` glob but name the source, e.g.

- *"…net loan charge-offs for **Wells Fargo**…"* → `wells_fargo_2024`
- *"…**Citigroup** Corporation…"* → `citigroup_2024`

The weak (string) pass cannot see these because it has no access to the gold doc_ids and
does no semantic matching. The strong pass joins each prompt to its gold doc and matches
the entity, so it surfaces leakage the string pass scores as zero.

---

## Notes per batch

- **batch_2 denominator.** 58 of 135 prompts are `refusal` (18), `ingest_only` (29),
  `extract_only` (6), `capability_gap` (4), `dispatcher_prompt` (1) — they have no gold
  answer doc, so they can't leak an answer location. Many *name* a file (because the task
  *is* "ingest/extract this file"), but that is not answer-location leakage. Hence
  72.7% of answerable vs 41.5% of all; the 72.7% is the fair figure.
- **batch_3 entity leaks are real.** Questions that name no file still leak via the
  company name in a one-filing-per-company corpus, e.g. *"What was **Apple's** net revenue
  in fiscal year 2022?"* → `APPLE_2022_10K`. The strong definition catches these; the weak
  one filed them under "no path".
- **batch_4 leakage is concentrated in finance** (one filing per company, and the
  questions name the company); HR and pharma docs are topical, so fewer prompts name the
  exact source. (Per `leakage_batch4.md`: finance ~80% / HR ~13% / pharma ~8%.)

## Bottom line

- The vidore single-doc batches (**batch_1 ~80%**, **batch_3 ~77%** of answerable
  prompts) are the most contaminated for retriever-vs-baseline testing — mostly explicit
  filenames plus some entity-name leaks.
- **batch_2 ~73%** of answerable prompts leak, again mostly via filenames.
- **batch_4 (corpus-level) is the cleanest at ~33%**, with leakage that is purely
  entity-name (never filenames) and concentrated in finance.
- Under the weak string-only definition the totals look far smaller (10% overall, 0% for
  batch_4) because it misses entity-name leakage entirely. **The strong definition (40.1%
  of answerable prompts) is the one to act on.**

## Artifacts
- `leakage_batch4.md`, `leakage_batch4.json` — the original batch_4 deep-dive (incl.
  recall cross-reference showing leaked prompts find the right file far more often).
- `/tmp/entity_leaks_all.json` — per-leak records for all four batches under the strong
  definition (`batch`, `eval_id`, `leak_kind`, `gold_doc`, `prompt`).
