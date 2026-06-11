# Functional test suite — EXTRACT: standalone audio (MP3 / WAV) transcription

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). This suite covers
the **EXTRACT** task *"Standalone Audio extraction (MP3 / WAV) — transcribe audio to text
chunks; produces time-aligned segments. English-only."*

Each test is a self-contained triple — a prompt, a per-case `data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and audio flags.

---

## The user task under test

> **JTBD: EXTRACT.** "Standalone Audio extraction (MP3 / WAV) — Transcribe audio to text
> chunks; **produces time-aligned segments**. **English-only**." — **P0**

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) `[multimedia]` extra + ffmpeg present; (2) ASR (`nvidia/parakeet-ctc-1.1b` local HF, or a remote Parakeet/Riva gRPC endpoint) loads; (3) ingest of a `.wav`/`.mp3` returns **non-empty time-aligned segments** — each row carries `source_id` + per-segment timing + segment text; (4) the transcript is queryable |
| Time | **medium** — single-clip transcription **≤ 2 min** (cold ASR load + a short clip) |
| Trigger rate | ≥ 95% — a "transcribe this audio / make my audio searchable" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <audio>` (auto-detected, **no `--input-type`**) then `retriever query`; segmentation rungs use `--segment-audio` / `--audio-split-type` / `--audio-split-interval` |
| Token usage | tracked, not gated |

Seed queries this suite is derived from:
- *"Transcribe meeting_clip.mp3 and tell me what the speaker is discussing."*
- *"Make my folder of MP3 podcasts searchable so I can find any mention of 'roadmap'."*
- *"Get the transcript out of customer_call_001.wav."*

---

## How the CLI transcribes audio (verified against the skill references + CLI + a live run)

Grounded by `skills/nemo-retriever/references/setup.md` ("Audio / video"),
`references/cli/ingest.md`, `docs/.../extraction/content-metadata.md`, and a **real local
ingest+query** of the WAV fixture in the project venv.

- **Auto-detect:** `retriever ingest <path>` detects audio from the extension — `.mp3` /
  `.wav` / `.m4a`. **`.flac` is silently filtered** (unsupported audio format). There is
  **no `--input-type`** flag on the shipped CLI; format is auto-detected.
- **Prereqs:** the `[multimedia]` extra **and** the ffmpeg host package. Without either,
  audio ingest fails. (Both verified present in this venv: ffmpeg, librosa, soundfile.)
- **ASR model:** `nvidia/parakeet-ctc-1.1b`, loaded locally via `[multimedia]` (HF) **or** a
  remote Parakeet/Riva **gRPC** endpoint. The `--dry-run` plan shows the ASR leg:
  `asr.audio_infer_protocol = grpc`, `asr.audio_endpoints`, `asr.segment_audio`, and an
  `audio` branch (`branch_summary: "audio:1"`).
- **Segmentation:** `--segment-audio` toggles ASR-side segmentation;
  `--audio-split-type size|time|frame` + `--audio-split-interval <int>` control chunking.
  The resolved plan reflects these as `audio.split_type` / `audio.split_interval`.
- **English-only:** Parakeet-CTC-1.1b is English-only and there is **no ASR language flag**
  (`--ocr-lang` selects the OCR-v2 language for *document images* only, not ASR). On
  non-English audio the library **silently emits garbage** — detecting non-English is the
  **skill layer's** job (filename/locale hint), covered by rung 5 / negative test (d).

### Where the timing lives (important for grading)

A **query** hit has exactly `{source, page_number, text}` — it does **not** expose timing.
Per-segment timing is asserted at the **extract / row-metadata layer**: each LanceDB row's
`metadata` JSON carries `{type: "audio", segment_start_seconds, segment_end_seconds}`, and
the row's `source` carries `{source_id, source_name}` = the audio path. (The library schema
names these `content_metadata.start_time` / `end_time` in **milliseconds**; the LanceDB CLI
row surfaces `segment_start_seconds` / `segment_end_seconds` in **seconds**.) Graders should
inspect the row metadata (e.g. open the LanceDB table) for the timing assertions.

### Live-run grounding (captured in this venv)

`retriever ingest data/multimodal_test.wav --segment-audio --audio-split-type time
--audio-split-interval 5` →

```
Ingested 1 file(s) → 6 row(s) in LanceDB <uri>/<table>.
```

6 **monotonically time-aligned** audio segments (each `type=audio`,
`source_id=...multimodal_test.wav`):

| # | start_s | end_s | transcript |
|---|---|---|---|
| 0 | 0.48 | 0.88 | Section one. |
| 1 | 1.12 | 2.56 | This is the first section of the document. |
| 2 | 4.00 | 4.96 | It has some more place |
| 3 | 5.59 | 7.27 | Text to show how the document looks like. |
| 4 | 8.55 | 9.75 | The text is not meant to be |
| 5 | 10.35 | 14.75 | Meaningful or informative, but rather to demonstrate the layout and formatting of the document. |

A follow-up `retriever query "What is the first section about?"` returned a JSON array of
`{source, page_number, text}` hits citing the `.wav` (top hit: *"This is the first section
of the document."*).

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## Fixtures

The catalog has **only a WAV** audio fixture: `data/multimodal_test.wav` (~16 s, 16 kHz).
Every case copies it (renamed to a realistic name). **There is no `.mp3` fixture.** The MP3
path through the CLI is **identical** — audio is auto-detected by extension and runs the same
ASR stage — so the `.wav` stands in for `.mp3` here; to exercise MP3 *specifically* a `.mp3`
fixture would need to be added to the catalog. The non-English cases reuse the same WAV bytes
but with a **non-English-locale filename** (`reunion_fr-FR.wav`), because the library has no
language detector — the guard must fire on the **name/locale hint**.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `extract-audio-001` | **Baseline.** One `.wav` → non-empty transcript rows; audio auto-detected. | `ingest`, `query` |
| 2 | `extract-audio-002` | **Time alignment.** Assert each row carries `segment_start_seconds`/`segment_end_seconds` (+ `type=audio`, `source_id`), increasing. | `ingest`, `query` |
| 3 | `extract-audio-003` | **Segmentation control.** `--segment-audio --audio-split-type time --audio-split-interval` re-chunks the clip; plan + segment count reflect it. | `ingest`, `query` |
| 4 | `extract-audio-004` | **Folder → searchable.** One ingest over a folder of audio (`audio:3`); a keyword query finds the mentioning segment, cited to its file. | `ingest`, `query` |
| 5 | `extract-audio-005` | **Acceptance + guard.** English clip → named index, queryable with timing citation; AND a non-English-named clip triggers the **English-only** guard (negative test d). | `ingest`, `query` |

The ladder: T1 proves ASR runs and emits text; T2 proves the segments are *time-aligned*;
T3 proves the agent can *control* the chunking; T4 proves a *corpus* of audio becomes
searchable; T5 composes the English happy path **and** the English-only negative guard into
the row's operational-pass gate.

---

### T1 — `extract-audio-001` · single WAV → transcript  *(complexity 1)*
- **Satisfies:** the core EXTRACT task, simplest form.
- **Data:** `data/meeting_clip.wav`.
- **Expected:** `RETRIEVER ingest data/meeting_clip.wav` (`branch_summary audio:1`) →
  `RETRIEVER query "What is the speaker discussing?" --top-k 5`. Rows non-empty; ≥ 1 hit
  cites the `.wav`. (Do not fabricate transcript content — assert non-empty + cited.)

### T2 — `extract-audio-002` · time-aligned segments  *(complexity 2)*
- **Satisfies:** the "produces time-aligned segments" clause.
- **Data:** `data/customer_call_001.wav`.
- **Adds:** a row-metadata assertion — each segment carries `segment_start_seconds` +
  `segment_end_seconds` (`type=audio`), non-negative and non-decreasing, with `source_id` =
  the audio path.

### T3 — `extract-audio-003` · segmentation control  *(complexity 3)*
- **Satisfies:** the segmentation flags directly.
- **Data:** `data/long_briefing.wav`.
- **Expected:** `RETRIEVER ingest data/long_briefing.wav --segment-audio
  --audio-split-type time --audio-split-interval 5` (`--dry-run` plan:
  `asr.segment_audio=true`, `audio.split_type=time`, `audio.split_interval=5`).
- **Adds:** explicit control of *how* the clip is chunked; changing the interval changes the
  segment count/boundaries (observed: 6 rows for the ~16 s clip at 5 s).

### T4 — `extract-audio-004` · folder → searchable  *(complexity 4)*
- **Satisfies:** "make my folder of audio searchable."
- **Data:** `data/` (`podcast_ep01/02/03.wav`).
- **Expected:** `RETRIEVER ingest data/` (`branch_summary audio:3`) → `RETRIEVER query
  "section" --top-k 10`; a hit containing the keyword cites one of the `.wav` files.
- **Adds:** folder-level ingest in one invocation + a keyword retrieval over the corpus.

### T5 — `extract-audio-005` · acceptance + English-only guard  *(complexity 5)*
- **Satisfies:** the full operational-pass row **plus** the English-only negative guard.
- **Data:** `data/town_hall_en.wav` (English) + `data/reunion_fr-FR.wav` (non-English name).
- **Expected (English):** `RETRIEVER ingest data/town_hall_en.wav --segment-audio
  --table-name audio_smoke` → `RETRIEVER query "town hall" --table-name audio_smoke
  --top-k 10`; answer cites `town_hall_en.wav` with a time-aligned segment.
- **Expected (non-English):** the skill detects the `fr-FR` locale hint and **warns** that
  audio ASR is English-only (Parakeet-CTC-1.1b, no language flag) instead of transcribing
  the French file and returning garbage.
- **Adds (the trap):** custom `--table-name` aligned across ingest + query, the time-aligned
  citation requirement, and the English-only guard.
- **Caveat (built into the test):** the library has no ASR language detector and no ASR
  language flag, so non-English detection is the skill layer's responsibility (filename/
  locale hint). If the skill omits the guard, the French file "passes" ingest but produces
  meaningless rows — the known failure mode this rung gates.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. Checks unique to this suite: **(a)** audio is
auto-detected (no `--input-type`); **(b)** transcript rows are non-empty and **time-aligned**
(`segment_start_seconds`/`segment_end_seconds` in row metadata, inspected at the LanceDB
layer — *not* in the query keys); **(c)** the folder rung ingests all files in one call;
**(d)** the English-only guard fires on the non-English-named clip.

**Note on live runs.** Audio ingest requires the **`[multimedia]` extra + ffmpeg** host
package, and only a **WAV** fixture is available (no `.mp3` — the MP3 path is identical and
would need a `.mp3` fixture to exercise specifically). The transcript text, segment count,
and timings above were captured from a real local ingest+query of `multimodal_test.wav` in
the project venv; the remaining expected outputs (folder `audio:3`, named-index alignment,
the English-only guard behavior) are grounded in the CLI source + `--dry-run` plan and the
skill references but have **not** been run live across all five cases. A full live run would
capture per-case row counts, ASR cold-start + per-clip latencies (the ≤ 2 min SLA), exact
transcript text for the renamed clips, and the skill's actual warning wording on the
non-English file.
