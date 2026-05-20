## Summary

<!-- NVBugs: Fixes XXXXXX. One paragraph — what was wrong in docs vs product/Helm. -->

## Files and page roles

| File | Bug(s) | What changed (one line) |
|------|--------|-------------------------|
| `docs/docs/extraction/...` | | |
| `nemo_retriever/docs/cli/README.md` | | |

**Deploy / Helm / deprecation detail only on:** `prerequisites-support-matrix.md` (and `nemo_retriever/helm/` if chart-related).

## Checklist (26.05 — Randy review)

- [ ] Bug expected result is met on the **owning** page(s), not duplicated across all touched files
- [ ] No `nimOperator`, `nvcr.io`, or "Helm chart ships…" prose on `multimodal-extraction.md` or package READMEs (unless bug is chart-only)
- [ ] `multimodal-extraction.md` is concepts only; chart/caption deploy detail is in `nemo_retriever/helm/README.md#charts-infographics-and-captioning-2605`
- [ ] Optional NIMs on support matrix: **one table**, no duplicate bullet list; Image captioning (26.05) = Omni guidance only (no repeated Omni + `nvcr.io` paragraph)
- [ ] **No mention** of `nemotron-nano-12b-v2-vl` / Nemotron Nano 12B VL anywhere in extraction docs or `nemo_retriever/helm/README.md` (Randy: removed; never 26.05 Helm; release notes only if documenting removal)
- [ ] Hosted OCR examples use `nemotron-ocr-v1` until v2 is published (not `nemotron-ocr-v2` in CLI/package README)
- [ ] No `graphic_elements` / yolox as required for 26.05 charts (`grep -i graphic nemo_retriever/helm/values.yaml` → 0)
- [ ] CLI examples: no orphaned flags/URLs for NIMs absent from 26.05 `values.yaml`
- [ ] `extract_charts` / `extract_infographics` documented as default **on** if mentioned
- [ ] Minimal diff — no unrelated pages unless in scope

## Test plan

- [ ] Grep published paths / source for removed misleading strings (e.g. old model as "default", removed NIM names)
- [ ] Subject-matter expert: page reads correctly **for that page's purpose** (conceptual vs matrix vs CLI)

## NVBugs

- Fixes: <!-- 6195023, 6195296 -->
