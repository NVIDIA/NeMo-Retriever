# Paste-in prompt for NVBugs / doc-drift fixes

Copy into Agent chat when fixing extraction documentation. Adjust NVBugs IDs and file list.

---

Fix NVBugs **XXXXXX** (and **YYYYYY** if batched). Edit **only** these files:

1. `docs/docs/extraction/prerequisites-support-matrix.md` — **only** file for Helm, `nimOperator.*`, hardware tables, Omni caption guidance, `nvcr.io` pins.
2. `docs/docs/extraction/multimodal-extraction.md` — pipeline behavior and model names; link to the matrix for deploy — **no** Helm flags or chart essays here.
3. `nemo_retriever/docs/cli/README.md` — CLI examples and short notes only.
4. `nemo_retriever/README.md` — same as CLI; link out for deploy detail.

**Do not**

- Add `nimOperator`, "Helm chart", or `nvcr.io` paths to (2), (3), or (4).
- Mention `nemotron-nano-12b-v2-vl` or Nemotron Nano 12B VL anywhere (Randy: never in 26.05 Helm docs; release notes only if documenting removal).
- Duplicate deploy or caption prose outside (1).
- Expand scope to unrelated pages (air-gap, deployment-options, faq) unless the bug lists them.

**Do**

- Minimal diff; satisfy the bug **Expected result** on the owning page; other pages get a sentence + link if needed.
- Verify against `nemo_retriever/helm/values.yaml` on branch `26.05` before claiming a NIM is in or out of the chart.

Open PR with template: `.github/PULL_REQUEST_TEMPLATE/doc_bug_fix.md`.

**26.05:** Agent must follow `.cursor/rules/nrl-26.05-release-docs.mdc` (critical errors table + pre-finish greps). Restart Agent chat after rule changes.
