# Synopsis — RETRIEVE: agentic retrieval, multi-turn (query rewrite + multi-turn)

**What user task this covers.** A user holds a **conversation** over a document and expects
the assistant to follow along: ask "what does the table show for James?", then just "and
Susan?", then "which of them scored higher in 2019?" — each follow-up leaning on what came
before. The user also expects the assistant to take a **vague** question and turn it into the
precise lookup it really means. Success means the agent **rewrites** each turn into a clear,
self-contained question and **remembers the context** across turns — returning the right
grounded answer every time.

**The key constraint.** The NeMo Retriever Library (NRL 26.05) is **stateless** — it has no
conversation or session memory, and its `retriever query` command takes one question at a
time with no way to pass prior turns. So multi-turn and query-rewrite are **not** features of
the retriever; they are things the **agent** must do: track the conversation itself, rewrite
each user turn into a standalone question (filling in "and Susan?" → "Susan's 2019 value",
resolving "that year", "the higher one", "whoever won"), and fire a fresh query each time.
These tests check that the **agent** does this correctly — not that the retriever remembers
anything.

**How we test it.** Five agent prompts, each a short conversation over a small real table
(`table_test.pdf`, whose values we read directly). We check the agent ingests the table with
table-structure on, then for **each turn** issues a correct **standalone** `retriever query`
whose wording resolves the pronouns/ellipsis from earlier turns — and that the final answers
match the real cell values (James 2019 = 978, Susan 2019 = 922, etc.).

**The five tests, simplest to hardest:**

1. **Single vague turn → rewrite.** One underspecified question ("his number for the 2019
   line?") that the agent must rewrite into a precise query naming James and 2019. No prior
   context yet — this isolates the rewrite skill. (Answer: 978.)
2. **Two-turn carry.** Turn 1 about James in 2019, then a bare "And Susan?" — only
   answerable by carrying "2019" from turn 1. The agent rewrites it to "Susan's 2019 value".
   (Answer: 922.)
3. **Three-turn comparison.** Adds "which of them scored higher that year?" — the agent must
   pull both 2019 cells and **compute** the comparison itself (James, 978 > 922).
4. **Reference resolution.** A chain where a later turn points at a *prior result* ("whoever
   scored higher that year") and at "that year"; a second document is loaded so even "the
   table" is a reference the agent must pin to the right file. (Resolves to Susan, then her
   2018 value, 976.)
5. **Acceptance gate.** One full conversation combining all of the above — a vague opener, an
   elliptical follow-up, an agent-computed comparison, and a derived-referent turn ("for
   whoever won, what did they have the year before?") — each emitted as a correct standalone
   query, with the agent explicitly noting that it (not the retriever) carried the context.

**Why this order.** Each rung adds exactly one new thing: first "rewrite one vague query,"
then "carry context to a second turn," then "compute a comparison across turns," then
"resolve a pronoun that points at a derived result," then everything composed into one
acceptance conversation. The time budget is **≤ 5 min** (longer than other RETRIEVE rows)
because each case is a multi-call sequence, not a single query.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run`) and the real
fixture values; **not yet run live**. Live runs need a reachable embedding/rerank backend
(hosted with an API key, or a local GPU) and the table-structure extraction backend. The
defining point — restated — is that **NRL is stateless and the multi-turn / query-rewrite
behavior lives entirely at the agent layer**. See `README.md` for the full spec and
`cases.json` for the machine-gradable definitions.
