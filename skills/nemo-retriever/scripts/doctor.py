#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the installed `retriever` engine satisfies the skill's contract.

Usage: <RETRIEVER_VENV>/bin/python skills/retriever/scripts/doctor.py
Exits 0 if all checks pass, 1 otherwise. Always runs a LIVE ingest+query probe.

The skill's one primitive is `retriever query --format evidence --hybrid` ->
{evidence, coverage}; this doctor gates on THAT invocation and result shape. `query`'s
DEFAULTS are unchanged (legacy `hits` output, vector-only) — `evidence`/`hybrid` are
opt-in flags the skill passes, so neither is gated here.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT_DIR = os.path.join(os.path.dirname(HERE), "contract")
EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
# Tiny self-contained probe doc, written to a temp corpus so the live ingest+query
# check needs no external fixture file.
PROBE_TEXT = (
    "Contract probe document.\n"
    "The capital of the test corpus is Probeville.\n"
    "This single short text file exists only so doctor.py can ingest one tiny "
    "document and run one query to assert the live hit schema.\n"
)

results = []  # (ok: bool, label: str, detail: str)


def check(ok, label, detail=""):
    results.append((bool(ok), label, detail))


def retriever_bin():
    return shutil.which("retriever")


def help_text(bin_path, subcmd):
    # Force a wide terminal so the rich/click help box does not truncate long
    # flag names (e.g. "--embed-model-na…"), which would break substring checks.
    env = dict(os.environ, COLUMNS="200")
    try:
        out = subprocess.run([bin_path, subcmd, "--help"], capture_output=True, text=True, timeout=60, env=env)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__ {e}"


def main():
    with open(os.path.join(CONTRACT_DIR, "cli-contract.json")) as _f:
        contract = json.load(_f)
    with open(os.path.join(CONTRACT_DIR, "query-result.schema.json")) as _f:
        rr_schema = json.load(_f)
    item_schema = rr_schema["$defs"]["evidence_item"]
    cov_schema = rr_schema["$defs"]["coverage"]

    bin_path = retriever_bin()
    check(
        bin_path is not None, "retriever CLI on PATH", "" if bin_path else "run skills/retriever/references/install.md"
    )
    if not bin_path:
        return report()

    # --- Required subcommands exist (static, no GPU) ---
    for sub in contract.get("subcommands_required", []):
        try:
            rc = subprocess.run([bin_path, sub, "--help"], capture_output=True, text=True, timeout=60).returncode
        except Exception:  # noqa: BLE001
            rc = 1
        check(rc == 0, f"subcommand `{sub}` exists")

    # --- query flag surface: required flags present (static). `query` is now both the
    # skill primitive and the power-user tool, so strategy knobs are allowed. ---
    rhelp = help_text(bin_path, "query")
    for flag in contract["query"]["required_flags"]:
        check(flag in rhelp, f"query has {flag}")

    # --- ingest flag surface (static, no GPU) ---
    ihelp = help_text(bin_path, "ingest")
    for flag in contract["ingest"]["required_flags"]:
        check(flag in ihelp, f"ingest has {flag}")
    for flag in contract["ingest"]["forbidden_flags"]:
        check(
            flag not in ihelp, f"ingest does NOT have {flag}", "engine changed: skill assumes single-pass auto-detect"
        )

    # --- Live probe: ingest tiny fixture, retrieve, validate result shape (GPU) ---
    tmp = tempfile.mkdtemp(prefix="retriever_doctor_")
    try:
        corpus = os.path.join(tmp, "corpus")
        os.makedirs(corpus)
        with open(os.path.join(corpus, "contract_probe.txt"), "w") as probe_f:
            probe_f.write(PROBE_TEXT)
        uri = os.path.join(tmp, "lancedb")
        table = "contract_probe"
        ing = subprocess.run(
            [
                bin_path,
                "ingest",
                corpus + "/",
                "--table-name",
                table,
                "--lancedb-uri",
                uri,
                "--embed-model-name",
                EMBED_MODEL,
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        check(ing.returncode == 0, "live ingest of fixture", ing.stderr.strip()[-300:])

        r = subprocess.run(
            [
                bin_path,
                "query",
                "What is the capital of the test corpus?",
                "--format",
                "evidence",
                "--top-k",
                "3",
                "--no-hybrid",
                "--table-name",
                table,
                "--lancedb-uri",
                uri,
                "--embed-model-name",
                EMBED_MODEL,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        check(r.returncode == 0, "live query --format evidence", r.stderr.strip()[-300:])
        result = None
        if r.returncode == 0:
            try:
                result = json.loads(r.stdout)
                check(
                    isinstance(result, dict) and "evidence" in result and "coverage" in result,
                    "query emits {evidence, coverage}",
                )
            except Exception as e:  # noqa: BLE001
                check(False, "query stdout is JSON", str(e))
        if isinstance(result, dict):
            ev = result.get("evidence")
            check(isinstance(ev, list) and len(ev) > 0, "query returned evidence")
            if isinstance(ev, list) and ev:
                ok, why = validate(ev[0], item_schema)
                check(ok, "evidence item matches query-result schema", why)
            cov = result.get("coverage")
            check(isinstance(cov, dict), "coverage is an object")
            if isinstance(cov, dict):
                ok, why = validate(cov, cov_schema)
                check(ok, "coverage matches query-result schema", why)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return report()


def validate(obj, schema):
    """Tiny dependency-free validator: required fields, types (incl. unions), enums."""
    if not isinstance(obj, dict):
        return False, "value is not an object"
    for req in schema.get("required", []):
        if req not in obj:
            return False, f"missing required field '{req}'"
    types = {
        "integer": int,
        "string": str,
        "number": (int, float),
        "object": dict,
        "array": list,
        "null": type(None),
        "boolean": bool,
    }
    for name, spec in schema.get("properties", {}).items():
        if name not in obj:
            continue
        if "type" in spec:
            allowed = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
            pytypes = []
            for key in allowed:
                mapped = types.get(key)
                if mapped is None:
                    continue
                pytypes.extend(mapped if isinstance(mapped, tuple) else [mapped])
            if pytypes and not isinstance(obj[name], tuple(pytypes)):
                return False, f"field '{name}' should be {spec['type']}, got {type(obj[name]).__name__}"
        if "enum" in spec and obj[name] not in spec["enum"]:
            return False, f"field '{name}'={obj[name]!r} not in {spec['enum']}"
    return True, ""


def report():
    failed = [r for r in results if not r[0]]
    for ok, label, detail in results:
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {label}"
        if detail and not ok:
            line += f"  -- {detail}"
        print(line)
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
