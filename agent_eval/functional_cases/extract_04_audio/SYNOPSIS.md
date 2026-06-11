# Synopsis — EXTRACT: standalone audio (MP3 / WAV) transcription

**What user task this covers.** A user has audio — a meeting clip, a customer call, a folder
of podcasts — and wants it turned into **text they can read and search**, broken into
**time-aligned segments** (each line tagged with where it falls in the recording) so they can
jump to or cite a moment. NeMo Retriever does this by auto-detecting `.mp3` / `.wav` audio
and running ASR (`nvidia/parakeet-ctc-1.1b`). The catch the task spells out: it is
**English-only** — there is no language switch, so non-English audio silently produces
nonsense, and the skill has to catch that.

**How we test it.** Five agent prompts, each handing the agent one or more audio files and
checking that the agent drives the `retriever` CLI correctly: `ingest` the audio (no
`--input-type` — format is auto-detected), get back **non-empty, time-aligned** transcript
segments, and `query` them. Success is operational (binary pass/fail), and a single clip
should transcribe in **under 2 minutes**.

**The five tests, simplest to hardest:**

1. **Single clip → transcript** — transcribe one `.wav` and get non-empty text rows back.
2. **Time-aligned segments** — confirm each transcript line carries its start and end time in
   the recording, so the user can jump to it.
3. **Segmentation control** — re-chunk a clip into short, fixed-length time segments using the
   audio split controls.
4. **Folder → searchable** — load a whole folder of recordings in one go and find any segment
   mentioning a keyword, cited back to its file.
5. **Acceptance + English-only guard** — transcribe an English clip into a named index and
   answer with a timestamped citation; **and** when handed a clip whose name signals a
   non-English language, the skill must **warn that audio is English-only** instead of
   transcribing it into garbage.

**Why this order.** Each rung adds exactly one thing: first "does ASR produce text," then "is
that text time-aligned," then "can the agent control the chunking," then "does a whole folder
become searchable," and finally the real gate — the English happy path *and* the English-only
safety guard together.

**Grounding note.** The English transcription path was **run live** in the project venv
against the one available WAV fixture: it produced 6 time-aligned segments and a working
query round-trip. There is **no MP3 fixture** in the catalog — the MP3 path is identical
(auto-detected), so the WAV stands in; a `.mp3` fixture would be needed to exercise MP3
specifically. Audio ingest also needs the `[multimedia]` extra plus ffmpeg.

**Status.** Tests are authored and the English happy path is grounded in a real local run;
the folder, named-index, and English-only-guard behaviors are grounded in the CLI source +
dry-run plan + skill references but **not yet run live** across all five cases. See
`README.md` for the full spec and `cases.json` for the machine-gradable definitions.
