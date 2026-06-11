# Functional test suite — EXTRACT: Standalone Video extraction (speech + on-screen text -> one searchable doc)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/nemo_retriever/src` (`retriever ingest` / `retriever query`).

This suite covers the **standalone video extraction** user task: take a video file
(MP4 / MOV / MKV / AVI), pull out **both** the spoken audio (ASR) **and** the on-screen
text (sampled frames -> page-elements + OCR), and land them as **one unified, searchable
document** per video. Each test is a self-contained triple — a prompt, a per-case `data/`
folder, and an expected output naming the correct `retriever` subcommand(s) and video flags.

---

## The user task under test

> **JTBD: EXTRACT — Standalone Video extraction (MP4 / MOV / MKV / AVI).** "Extract speech +
> on-screen text from video files into a unified searchable document." — **P0**

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) `retriever ingest <video>` auto-detects the format, (2) audio demuxed + transcribed (speech rows), (3) sampled frames OCR'd into on-screen-text rows tagged `content_type=VIDEO_FRAME`, (4) speech + frame rows unified under ONE `source_id` (the video path) with timestamp lineage, (5) a query surfaces content from both legs |
| Time | **slow — full extraction (speech + on-screen) ≤ 10 min** on a 5-min clip (bundled fixture is ~10 s, so live runs are far under budget) |
| Trigger rate | ≥ 95% — a "transcribe / make this video searchable / get spoken + on-screen text" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <video>` with the right video flags then `retriever query`; **no** `--input-type` |
| Token usage | tracked, not gated |

Seed queries this suite is derived from:
- *"Transcribe aurora_townhall_excerpt.mp4 and tell me what the speaker's topic was."*
- *"Make every video in this folder searchable."*
- *"I need both spoken dialogue and any text shown on screen from this video — searchable as one thing."*

---

## How the CLI handles video (grounded against `cli/main.py` + offline `--dry-run`)

`retriever ingest <path>` routes by **file extension** — there is **no `--input-type`**.
A `.mp4` (and `.mov`/`.mkv`/`.avi`) routes to the **`video`** family; `--dry-run` (offline)
prints `branch_summary: "video:1"`. Verified against the bundled fixture:

- **Audio (ASR / speech) leg.** `audio.enabled: true` by default. The audio track is
  demuxed (ffmpeg) and transcribed by **`nvidia/parakeet-ctc-1.1b`** into speech rows.
  Toggle with `--video-extract-audio` / `--no-video-extract-audio`; segmentation via
  `--segment-audio`, `--audio-split-type size|time|frame`, `--audio-split-interval`.
- **Frame (on-screen text / OCR) leg.** `video_frames.enabled: true` by default,
  `fps: 0.5`, perceptual dedup on (`dedup: true`, hamming 5, max-dropped 2), plus
  OCR-text dedup (`video_frame_text_dedup.enabled: true`, `max_dropped_frames: 2`).
  Sampled frames go through **`nemotron-page-elements-v3` -> `nemotron-ocr-v2`**; the
  resulting on-screen-text rows are tagged **`content_type=VIDEO_FRAME`**. Toggle with
  `--video-extract-frames` / `--no-video-extract-frames`; tune with `--video-frame-fps`,
  `--video-frame-dedup`, `--video-frame-text-dedup`,
  `--video-frame-text-dedup-max-dropped-frames`.
- **Unification.** Both legs are indexed against the **same `source_id`** (the video path),
  each row carrying **timestamp lineage**. `--video-av-fuse` (`audio_visual_fuse.enabled:
  true` by default) fuses the audio/visual streams so a single video is one searchable
  document.

Grounding from `--dry-run` (offline, no network) on the fixture:

```
# default
ingest aurora_townhall_excerpt.mp4 --dry-run
  -> branch_summary "video:1"
     audio.enabled true (split_type size, split_interval 500000)
     audio_visual_fuse.enabled true
     video_frames.enabled true, fps 0.5, dedup true (hamming 5, max_dropped 2)
     video_frame_text_dedup.enabled true, max_dropped_frames 2

# frames off (rung 1)
ingest aurora_townhall_excerpt.mp4 --video-extract-audio --no-video-extract-frames --dry-run
  -> video_frames.enabled false; audio.enabled true; audio_visual_fuse.enabled true
```

**Install prerequisite.** Video/audio needs the **`.[multimedia]`** extra **and** the system
**`ffmpeg`** binary on PATH (used to demux the audio track and sample frames). Without both,
the video branch cannot demux/sample.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

**Content grounding.** The bundled fixture is `aurora_townhall_excerpt.mp4` (~10 s, 141 KB) —
the only video in the catalog. We do **not** assert the transcript wording or the on-screen
text verbatim (not verified). Tests assert **structural** behavior: speech rows + VIDEO_FRAME
rows unified under one `source_id`, with timestamp lineage, surfaced by a query.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `extract-video-001` | **Baseline — speech only.** One MP4 auto-detected; audio demuxed + transcribed (ASR). Frames OFF (`--no-video-extract-frames`) to isolate the speech leg. | `ingest`, `query` |
| 2 | `extract-video-002` | **Add frame OCR.** `--video-extract-frames` -> sampled frames -> page-elements + OCR -> on-screen-text rows tagged `VIDEO_FRAME`. | `ingest`, `query` |
| 3 | `extract-video-003` | **Unify.** Speech rows + VIDEO_FRAME rows share ONE `source_id` with timestamp lineage; a query surfaces content from BOTH legs (folder ingest + `--video-av-fuse`). | `ingest`, `query` |
| 4 | `extract-video-004` | **Sampling / dedup controls.** Tune `--video-frame-fps 1.0` + `--video-frame-text-dedup` so repeated on-screen text isn't re-indexed; verify in `--dry-run`. | `ingest --dry-run`, `ingest`, `query` |
| 5 | `extract-video-005` | **Acceptance gate.** Fresh named index; both legs extracted; a spoken phrase AND an on-screen phrase each surfaced with timestamp citation; ≤ 10 min. | `ingest`, `query` |

The ladder: T1 proves the audio (speech) leg alone; T2 turns on the visual (on-screen-text)
leg; T3 proves the two legs are **one** searchable document under a single `source_id`; T4
adds operator control of the frame-sampling/dedup knobs; T5 composes everything into the
row's real operational-pass / acceptance gate (both legs, citations + timestamps, ≤ 10 min).

---

### T1 — `extract-video-001` · single MP4 -> speech rows (frames off)  *(complexity 1)*
- **Satisfies:** the EXTRACT-video task, audio leg only.
- **Data:** `data/aurora_townhall_excerpt.mp4`.
- **Expected:** `RETRIEVER ingest data/aurora_townhall_excerpt.mp4 --video-extract-audio
  --no-video-extract-frames` (auto-detected video, no `--input-type`) -> `RETRIEVER query
  "What is the speaker's topic?" --top-k 5`. ≥ 1 speech hit citing the mp4 source; no
  VIDEO_FRAME rows (frames off). Topic wording not asserted verbatim.

### T2 — `extract-video-002` · add frame OCR -> VIDEO_FRAME rows  *(complexity 2)*
- **Satisfies:** the on-screen-text half of the task.
- **Data:** `data/aurora_townhall_excerpt.mp4`.
- **Adds:** `--video-extract-frames` -> sampled frames -> page-elements-v3 -> ocr-v2;
  ≥ 1 row tagged `content_type=VIDEO_FRAME`, surfaced by query; speech rows still present.

### T3 — `extract-video-003` · unified doc, both legs, one source_id  *(complexity 3)*
- **Satisfies:** the task's core promise — one unified searchable document per video.
- **Data:** `data/` (single-video folder; see caveat).
- **Expected:** `RETRIEVER ingest data/ --video-extract-audio --video-extract-frames
  --video-av-fuse` -> a query that returns BOTH a speech hit and a VIDEO_FRAME hit, all
  under the one mp4 `source_id`, with timestamp lineage.
- **Caveat:** the catalog has a single video fixture, so `data/` is a one-video folder; the
  folder-ingest path (`branch_summary video:N`) is exercised with N=1.

### T4 — `extract-video-004` · frame sampling / dedup controls  *(complexity 4)*
- **Satisfies:** the task plus operator control of the frame-OCR leg.
- **Data:** `data/aurora_townhall_excerpt.mp4`.
- **Expected:** `--dry-run` first confirms `video_frames.fps == 1.0`, `video_frames.dedup ==
  true`, `video_frame_text_dedup.max_dropped_frames == 3`; then the real ingest with
  `--video-frame-fps 1.0 --video-frame-dedup --video-frame-text-dedup
  --video-frame-text-dedup-max-dropped-frames 3`. Repeated on-screen text is collapsed (no
  flood of duplicate VIDEO_FRAME rows); query still surfaces the on-screen text.

### T5 — `extract-video-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass row + the ≤ 10 min (slow) validation path.
- **Data:** `data/aurora_townhall_excerpt.mp4`.
- **Expected:** ingest into `--table-name video_smoke` (both legs + `--video-av-fuse`) ->
  one query for a spoken phrase and one for an on-screen phrase, each `--table-name
  video_smoke`, each returning a hit citing the mp4 `source_id` with timestamp lineage.
- **Adds (the gate):** custom `--table-name` aligned across ingest AND query, both-legs
  assertion, timestamp-citation assertion, and the ≤ 10 min slow-bucket deadline.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. Checks unique to this suite: **(a)** the `.mp4` is
auto-detected as video (no `--input-type`); **(b)** the speech leg (parakeet ASR) and the
frame leg (page-elements + ocr, `content_type=VIDEO_FRAME`) both produce rows; **(c)** those
rows are unified under one `source_id` with timestamp lineage; **(d)** a query surfaces
content from both legs.

**Note on live runs (not run live).** These tests are authored and **not yet run live**.
Expected outputs are grounded in the CLI source and offline `--dry-run` plans, not a live
ingest/query. A live run additionally requires the **`.[multimedia]`** extra **and** the
system **`ffmpeg`** binary on PATH (to demux audio + sample frames); the frame-OCR and
embedding legs may call hosted NIMs on a CPU-only host. A live run would capture real row
counts (speech vs. VIDEO_FRAME), the actual transcript/on-screen wording, timestamp lineage,
extraction latency, and token baselines, and would let the T1/T2/T3 prompts be re-grounded
on the fixture's real spoken topic and on-screen phrases. The catalog ships **one** MP4
fixture; the other container formats **`.mov`/`.mkv`/`.avi`** route through the **identical**
`video` branch but are **not** present in the catalog, so they are not exercised here.
