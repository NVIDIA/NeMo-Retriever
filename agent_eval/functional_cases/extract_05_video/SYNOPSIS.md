# Synopsis — EXTRACT: standalone video (speech + on-screen text -> one searchable doc)

**What user task this covers.** A user has video files (MP4 / MOV / MKV / AVI) and wants
them made **searchable** — capturing **both** what is *said* (the spoken audio) **and** what
is *shown* (text on slides, titles, captions burned into the frames), folded into **one
searchable document per video**. Success means: the library ingests the video, demuxes and
transcribes the audio, samples and OCRs the frames, and lands both kinds of text against the
same video so a single search can find content from either — all within ~10 minutes for a
5-minute clip.

**How we test it.** Five agent prompts, each handing the agent the bundled video fixture
(`aurora_townhall_excerpt.mp4`, a ~10 s excerpt) and checking that the agent drives the
`retriever` CLI correctly: `retriever ingest <video>` (the format is **auto-detected** — no
`--input-type`), the right **video flags** for the speech leg (`--video-extract-audio`) and
the on-screen-text leg (`--video-extract-frames`, with `--video-frame-fps` / dedup controls),
and a `retriever query` that surfaces the result. Because we have not run the fixture live,
the tests assert **structural** behavior — speech rows plus on-screen-text rows (tagged
`VIDEO_FRAME`) unified under one video `source_id`, with timestamp lineage — rather than the
exact transcript or on-screen wording.

**The five tests, simplest to hardest:**

1. **Speech only** — one MP4 auto-detected; demux + transcribe the audio. Frames OFF, so
   this isolates the spoken-words leg.
2. **Add frame OCR** — turn on frame sampling so the on-screen text is OCR'd into rows
   tagged `VIDEO_FRAME`, alongside the speech rows.
3. **Unify** — prove the speech rows and the on-screen-text rows are **one** document: same
   video `source_id`, timestamp lineage, and a single search that finds content from both.
4. **Sampling / dedup controls** — tune the frame rate and collapse repeated on-screen text
   so a static slide isn't indexed over and over; confirm the plan before running.
5. **Acceptance gate** — a clean end-to-end run into a custom-named index that extracts both
   legs and, in under 10 minutes, surfaces both a spoken phrase and an on-screen phrase, each
   with a timestamp and citing the video. This is the test the others build up to.

**Why this order.** Each rung adds exactly one thing: first the audio (speech) leg, then the
visual (on-screen-text) leg, then the proof the two are unified into one searchable document,
then operator control of how frames are sampled/deduped, then everything composed into the
real pass/fail gate with timestamps, citations, and the time budget.

**Prerequisite.** Live video extraction needs the `.[multimedia]` extra **and** the system
`ffmpeg` binary (to demux audio and sample frames).

**Status.** Tests are authored and grounded in the real CLI plus offline `--dry-run` plans;
**not yet run live** (a live run needs `[multimedia]` + ffmpeg and would also re-ground the
prompts on the fixture's real spoken topic and on-screen text). The catalog ships **one** MP4
fixture; `.mov`/`.mkv`/`.avi` route through the identical video branch but aren't in the
catalog. See `README.md` for the full spec and `cases.json` for the machine-gradable
definitions.
