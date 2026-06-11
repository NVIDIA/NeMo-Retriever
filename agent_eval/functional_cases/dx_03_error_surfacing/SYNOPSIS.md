# Synopsis — DX: error surfacing (clear, machine-parseable failures)

**What user task this covers.** When something goes wrong, a developer needs the retriever to
**fail loudly but cleanly**: a clear, machine-parseable error message — not a wall of Python
stack trace — that says *what* broke and *whether it's worth retrying*. This row covers the
four failure modes the spec calls out: an **unsupported file**, an **embedding rate limit**,
an **unavailable vector DB / index**, and a **malformed query**. Success means each one comes
back as a tidy, actionable error, with **retry/backoff guidance where it makes sense**
(rate limits, transient network blips) and **permanent failures labeled distinctly** from
transient ones.

**How we test it.** Five agent prompts, each asking the agent to *deliberately trigger* a
failure and report the exact error. We check the agent drives the `retriever` CLI to the
right failure (`ingest` an unsupported `.eml`; `query` a table that doesn't exist; `query`
with a bad `--top-k`), and that the result is a clean `Error: <message>` on stderr with a
sensible exit code — never a bubbled traceback. The structured-error contract is grounded in
the repo's `api/util/exception_handlers/` (the extract layer attaches a
`{task, status, source_id, error_msg}` metadata tag instead of crashing) and the NIM client's
retry/backoff logic (HTTP 429 → up to 5 retries with exponential backoff).

**The five tests, simplest to hardest:**

1. **Unsupported format** — try to load `sample.eml`; get back
   `Error: Unsupported input file type(s) for retriever ingest: …sample.eml` (exit 1). A
   *permanent* error, named clearly, no stack trace. The floor.
2. **Index unavailable** — query a table that doesn't exist; get
   `Error: Table 'does_not_exist' was not found` (exit 1). Moves the failure to the
   query / vector-DB layer.
3. **Malformed query** — bad flag values, from two layers: `--top-k 0` trips the option
   parser (`Invalid value for '--top-k': 0 is not in the range x>=1.`, exit 2), while
   `--candidate-k 2 --top-k 5` trips an SDK check
   (`candidate_k (2) must be greater than or equal to top_k (5).`, exit 1). Each names the
   flag and the rule it broke.
4. **Transient vs permanent** — contrast a *rate-limit* error (HTTP 429 from the embedding
   service → retried up to 5 times with exponential backoff; retry guidance is built in)
   against the *permanent* unsupported-file error (fail fast, no retry). Adds the
   transient/permanent taxonomy and the retry/backoff semantics.
5. **Acceptance gate** — induce a failure at every stage in one pass (ingest/format,
   query/vDB, validation), prove each is clean and machine-parseable, label each
   permanent or transient, and cite how the service layer structures errors. This is the
   test the others build up to.

**Why this order.** Each rung adds exactly one new thing: first "does an induced failure
surface cleanly at all," then a different pipeline layer (query/vDB), then caller-input
validation across two error layers, then the transient-vs-permanent taxonomy with
retry/backoff, then everything composed — failures at every stage, each parseable, labeled,
with retry guidance only where it belongs.

**Status.** Tests are authored and grounded in the real CLI and source. The four CLI errors
in rungs 1–3 (and the permanent side of rungs 4–5) were **induced live, offline** against the
venv binary — the messages and exit codes in `cases.json` are the real captured output. The
**rate-limit (HTTP 429) retry/backoff** path in rungs 4–5 is **grounded in source**
(`models/nim/primitives/nim_client.py`) and **not run live**, because a real 429 needs a
live, throttled hosted endpoint and an API key (and may bill). See `README.md` for the full
spec and `cases.json` for the machine-gradable definitions.
