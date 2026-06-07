#!/usr/bin/env python3
"""Phase 4 (functional): merge one-or-more functional_eval.json files into a report.

Usage:
  functional_report.py --eval claude=<run>/functional_eval.json --eval codex=<run>/functional_eval.json \
                       [--out functional_report.md]

Renders: overall pass-rate per agent, a pass-rate matrix by functional_type x agent, and a
failing-IDs appendix with reasons. Separates *infra/grading* non-results (run_timeout,
run_error, judge_error, no_criteria) from genuine logic FAILs so a setup outage isn't read as
a low pass-rate. Pure stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import gitinfo  # noqa: E402

_NONRESULT = {"run_timeout", "run_error", "judge_error", "no_criteria"}


def _run_commit(doc: dict) -> dict | None:
    """The code commit recorded when the eval RAN (run_config.code_commit), if present."""
    rd = doc.get("run_dir")
    if not rd:
        return None
    cfg = Path(rd) / "run_config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text()).get("code_commit")
    except Exception:  # noqa: BLE001
        return None


def _load(spec: str) -> tuple[str, dict]:
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = Path(path).resolve().parent.name
    doc = json.loads(Path(path).read_text())
    return label, doc


def _pct(p, n):
    return f"{100*p/n:.0f}%" if n else "—"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--eval", action="append", required=True, dest="evals", help="label=path/to/functional_eval.json (repeatable)."
    )
    ap.add_argument("--out", type=Path, default=Path("functional_report.md"))
    args = ap.parse_args(argv)

    runs = [_load(s) for s in args.evals]
    if not runs:
        print("no eval files", file=sys.stderr)
        return 1

    types = sorted({q["functional_type"] for _, d in runs for q in d["per_query"]})
    L = []
    L.append("# Functional (corpus-variants) evaluation — pass/fail\n")
    L.append(
        "Single-turn behavioral tests (`functional_corpus_variants`), graded pass/fail. "
        "`retrieval_answer`/`ingest_plus_answer` use an LLM PASS/FAIL rubric vs the manifest's "
        "`validation_signal`+`expected_output_shape`; `ingest` adds a programmatic LanceDB "
        "rows>0 gate. Stateful tests are out of scope (separate driver).\n"
    )

    # Code version — the commit of the code that RAN each eval (from run_config.code_commit),
    # falling back to the report-time HEAD for runs that predate commit-recording.
    L.append("## Code version\n")
    report_info = gitinfo.git_commit()
    any_run_commit = False
    L.append("| run | eval-time code commit |")
    L.append("|---|---|")
    for label, d in runs:
        ci = _run_commit(d)
        if ci:
            any_run_commit = True
            L.append(f"| {label} | {gitinfo.format_commit(ci)} |")
        else:
            L.append(f"| {label} | _not recorded — predates commit logging_ |")
    L.append(
        f"\nReport generated from working tree at {gitinfo.format_commit(report_info)}."
        + (
            ""
            if any_run_commit
            else " ⚠️ No run recorded its eval-time commit; the line above is the **report-time** "
            "code version, which may differ from what produced these runs."
        )
    )
    L.append("")

    # Overall per agent.
    L.append("## Overall\n")
    L.append("| agent | model | PASS | gradable | pass-rate | non-results |")
    L.append("|---|---|---|---|---|---|")
    for label, d in runs:
        o = d["summary"]["overall"]
        nonres = sum(c for s, c in d["summary"]["status_counts"].items() if s in _NONRESULT)
        L.append(
            f"| {label} | {d.get('model') or '?'} | {o['passed']} | {o['gradable']} | "
            f"{_pct(o['passed'], o['gradable'])} | {nonres} |"
        )
    L.append("")

    # Matrix by functional_type x agent.
    L.append("## Pass-rate by functional_type\n")
    L.append("| functional_type | " + " | ".join(label for label, _ in runs) + " |")
    L.append("|---" * (len(runs) + 1) + "|")
    for ft in types:
        cells = []
        for _, d in runs:
            r = d["summary"]["by_functional_type"].get(ft)
            cells.append(f"{r['passed']}/{r['gradable']} ({_pct(r['passed'], r['gradable'])})" if r else "—")
        L.append(f"| {ft} | " + " | ".join(cells) + " |")
    L.append("")

    # Non-results breakdown (infra/grading, not logic).
    L.append("## Non-results (infra / grading, excluded from pass-rate)\n")
    any_nonres = False
    for label, d in runs:
        bad = {s: c for s, c in d["summary"]["status_counts"].items() if s in _NONRESULT}
        if bad:
            any_nonres = True
            L.append(f"- **{label}**: " + ", ".join(f"{s}={c}" for s, c in sorted(bad.items())))
    if not any_nonres:
        L.append("_None — every query produced a gradable result._")
    L.append("")

    # Failing IDs appendix.
    L.append("## Failing queries (logic FAIL)\n")
    for label, d in runs:
        fails = [q for q in d["per_query"] if q["status"] == "ok" and not q["passed"]]
        L.append(f"### {label} — {len(fails)} fails\n")
        if not fails:
            L.append("_None._\n")
            continue
        L.append("| query_id | type | gate | judge | reason |")
        L.append("|---|---|---|---|---|")
        for q in sorted(fails, key=lambda x: (x["functional_type"], x["query_id"])):
            gate = (
                "—"
                if q.get("gate_pass") is None
                else ("ok" if q["gate_pass"] else f"FAIL(rows={q.get('lancedb_rows')})")
            )
            judge = q.get("judge_verdict") or "—"
            reason = (q.get("reason") or "").replace("|", "\\|")[:160]
            L.append(f"| `{q['query_id']}` | {q['functional_type']} | {gate} | {judge} | {reason} |")
        L.append("")

    args.out.write_text("\n".join(L))
    print(f"Wrote {args.out}  ({len(runs)} run(s), {len(types)} functional types)")
    for label, d in runs:
        o = d["summary"]["overall"]
        print(f"  {label}: PASS {o['passed']}/{o['gradable']} = {o['pass_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
