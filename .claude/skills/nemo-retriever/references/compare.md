# retriever compare

Comparison utilities. Optional subcommands are registered lazily — if the
relevant module is installed, you'll see:

- `retriever compare json` — diff two JSON files (extraction sidecars, eval
  outputs, recall outputs).
- `retriever compare results` — diff two retrieval/eval result bundles.

Run `retriever compare --help` to see which subcommands are present in your
install.

## When to use this

- You changed an extraction flag, ran the pipeline twice, and want a
  semantic diff of the outputs (not a textual diff).
- You ran [[recall]] or [[eval]] twice and want to know which queries
  regressed / improved.

**Use a different command when:**

- You want a single-number metric, not a diff → [[recall]] / [[eval]].
- You want a UI / portal for sweep comparison → [[harness]] (`portal` /
  `compare`).

## Canonical invocations

```bash
retriever compare json before.json after.json
retriever compare results runs/baseline/ runs/candidate/
```

Run `--help` on each subcommand for the exact flag set; the modules are
optional and may expose different options across releases.

## Common failure modes

- **`retriever compare json` not found** — the `compare_json` module isn't
  installed. Install the extras (or upgrade the package).
- **Diff shows everything different** — files have non-stable key order or
  embedded timestamps; the subcommand normalises common cases but not all.

## Related

- [[recall]] / [[eval]] — produce the artifacts this command compares.
- [[harness]] `compare` — session-level comparison with summaries.
