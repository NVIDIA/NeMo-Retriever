# Remove Graphic Elements Model — Design

**Date:** 2026-05-04
**Branch:** `edwardk/remove-graphic-elements`

## Goal

Fully deprecate the optional Nemotron Graphic Elements V1 model and remove all references from the nemo-retriever pipeline, infrastructure, tests, and docs. After this change, chart text extraction is performed by the OCR actor alone; the OCR text is emitted verbatim (no rearrangement, no `-` separators).

## Background

The graphic-elements model is an object detector that identifies structural regions inside a chart crop (chart_title, x_title, y_title, xlabel, ylabel, legend_title, legend_label, mark_label, value_label, other). Today it is used in two places:

1. **OCR actors** (`nemo_retriever/src/nemo_retriever/ocr/`) — when `use_graphic_elements=True`, the OCR actor invokes graphic-elements on each chart crop in addition to OCR, then calls `join_graphic_elements_and_ocr_output()` to merge the two outputs into a structured chart string.
2. **Standalone chart stage** (`nemo_retriever/src/nemo_retriever/chart/`) — `GraphicElementsActor` / `GraphicElementsCPUActor` re-do the same combined graphic-elements + OCR work as a separate pipeline stage. This stage is the only place where the chart actor itself talks to OCR.

The graphic-elements model only rearranges words and inserts separators. With it removed, chart text is the OCR output as-is.

## Non-goals

- Removing chart **detection**. Page-level chart bbox detection in `nemotron-page-elements-v3` is unchanged. The `chart` content type continues to appear in document metadata.
- Refactoring OCR, infographic, or table extraction beyond the graphic-elements removal.
- Cleaning up unrelated tech debt encountered along the way.

## Architecture after the change

```
page_elements_v3  ──► detects chart/table/infographic regions
        │
        ▼
OCR actor  ──► crops charts/tables/infographics, runs OCR
              emits `chart`, `infographic`, `table` columns
              chart text = OCR text in raw word order
```

No standalone chart actor. No graphic-elements model service. No chart-actor → OCR coupling.

## Changes by area

### 1. Delete the graphic-elements model

- `nemo_retriever/src/nemo_retriever/model/local/nemotron_graphic_elements_v1.py` — delete file
- `nemo_retriever/src/nemo_retriever/model/local/__init__.py` — drop `NemotronGraphicElementsV1` from `__all__` and the `__getattr__` dispatch
- `nemo_retriever/pyproject.toml` — drop the `nemotron-graphic-elements-v1` dependency

### 2. Delete the standalone chart pipeline stage

- `nemo_retriever/src/nemo_retriever/chart/` — delete the entire module (chart_detection.py, cpu_actor.py, gpu_actor.py, shared.py, config.py, commands.py, __init__.py)
- `nemo_retriever/src/nemo_retriever/application/pipeline/stage_registry.py` — drop the `chart_graphic_elements` stage entry
- `nemo_retriever/src/nemo_retriever/application/pipeline/build_plan.py` — drop graphic-elements stage wiring
- Any CLI plumbing in `nemo_retriever/src/nemo_retriever/pipeline/__main__.py` for the chart stage — remove

### 3. Strip graphic-elements from OCR actors

In `nemo_retriever/src/nemo_retriever/ocr/shared.py`:
- Remove `use_graphic_elements` parameter from `*_ocr_for_table_chart_infographic*` functions (both pandas and the parse variant)
- Remove `_match_graphic_elements_for_chart` helper and any `getattr(row, "graphic_elements_v1", ...)` reads
- Delete the `if label_name == "chart" and use_graphic_elements:` branches (lines 632–646 and 700–705); chart label rows always fall through to the plain-OCR `chart_items.append(entry)` path

In `nemo_retriever/src/nemo_retriever/ocr/{cpu_ocr,gpu_ocr,cpu_ocrv2,gpu_ocrv2}.py`:
- Drop `self.ocr_kwargs["use_graphic_elements"] = ...` initializations
- Drop any constructor parameter / config-key handling for `use_graphic_elements`

### 4. Delete the rearrangement helper

`nemo_retriever/src/nemo_retriever/utils/table_and_chart.py` — delete `join_graphic_elements_and_ocr_output()` and any helpers used only by it. Keep table-related helpers.

### 5. API-side cleanup

- `api/src/nv_ingest_api/internal/primitives/nim/model_interface/yolox.py` — delete the `YoloxGraphicElementsModelInterface` class and the `YOLOX_GRAPHIC_*` constants used only by it (keep `YOLOX_GRAPHIC_MIN_SCORE` if referenced by surviving table/page interfaces; otherwise delete)
- `api/api_tests/internal/primitives/nim/model_interface/test_yolox_interface_graphic_elements.py` — delete file
- `api/src/nv_ingest_api/internal/extract/image/chart_extractor.py` — drop graphic-elements wiring; chart text now consumed from the OCR-produced `chart` column
- `api/src/nv_ingest_api/util/image_processing/table_and_chart.py` — drop the join helper and any graphic-elements bbox-matching utilities

### 6. Top-level pipeline impls

- `src/nv_ingest/pipeline/default_pipeline_impl.py`
- `src/nv_ingest/pipeline/default_libmode_pipeline_impl.py`
- `src/nv_ingest/framework/orchestration/ray/examples/pipeline_test_harness.py`
- `src/nv_ingest/api/v1/health.py`

Strip every reference to `graphic_elements_invoke_url`, `use_graphic_elements`, the chart actor, and the graphic-elements health probe.

### 7. Params, harness, local stages

- `nemo_retriever/src/nemo_retriever/params/models.py` — drop graphic-elements model entry
- `nemo_retriever/src/nemo_retriever/harness/config.py`, `harness/run.py`, `harness/portal/app.py` — drop graphic-elements URL/options
- `nemo_retriever/src/nemo_retriever/local/stages/stage4_chart_extractor.py` — rewrite to OCR-only chart output (or delete if it became a thin shim)
- `nemo_retriever/src/nemo_retriever/local/stages/stage999_post_mortem_analysis.py` — drop graphic-elements counters/timings
- `nemo_retriever/src/nemo_retriever/utils/ray_resource_hueristics.py` — drop graphic-elements GPU-mem heuristic
- `nemo_retriever/src/nemo_retriever/text_embed/text_embed.py` — strip references if present
- `nemo_retriever/src/nemo_retriever/graph/multi_type_extract_operator.py`, `graph/ingestor_runtime.py` — drop chart-actor stage references
- `nemo_retriever/src/nemo_retriever/infographic/__init__.py`, `infographic/infographic_detection.py` — verify the references are import-only / string labels and remove; do not gut infographic functionality

### 8. Harness tooling

- `tools/harness/src/nv_ingest_harness/cases/graphic_elements.py` — delete file
- `tools/harness/src/nv_ingest_harness/cli/run.py`, `service_manager/docker_compose.py`, `service_manager/helm.py` — drop graphic-elements service / case / URL references
- `tools/harness/pyproject.toml` — drop graphic-elements-related deps if present
- `tools/harness/README.md`, `tools/harness/plans/SERVICE_MANAGER.md` — prune graphic-elements paragraphs

### 9. Configuration

- `config/default_pipeline.yaml`
- `config/custom_summarization_pipeline.yaml`
- `nemo_retriever/src/nemo_retriever/ingest-config.yaml`

Drop `graphic_elements_invoke_url`, `use_graphic_elements`, the chart-graphic-elements stage block, and any `nemotron-graphic-elements-v1` model entries.

### 10. Infrastructure

- `helm/templates/nemotron-graphic-elements-v1.yaml` — delete
- `helm/values.yaml` — remove the `nemotron-graphic-elements-v1` block + URL pointing at it
- `helm/mig/nv-ingest-mig-values.yaml`, `helm/mig/nv-ingest-mig-values-25x.yaml` — drop graphic-elements override blocks
- `helm/overrides/values-{a10g,a100-40gb,l40s,rtx-pro-4500}.yaml` — drop graphic-elements override blocks
- `helm/README.md` — prune graphic-elements references
- `docker-compose.yaml`, `docker-compose.a10g.yaml`, `docker-compose.a100-40gb.yaml`, `docker-compose.l40s.yaml`, `docker-compose.rtx-pro-4500.yaml` — delete the `graphic-elements` service block; remove env-var references in dependent services
- `ci/scripts/validate_deployment_configs.py` — drop graphic-elements from the expected-services allowlist
- `.github/workflows/integration-test-library-mode.yml` — remove graphic-elements service / URL setup
- `.github/workflows/huggingface-nightly.yml` — remove graphic-elements job / artifact step

### 11. Tests

Delete:
- `nemo_retriever/tests/test_chart_graphic_elements.py`
- `api/api_tests/internal/primitives/nim/model_interface/test_yolox_interface_graphic_elements.py`
- `tools/harness/src/nv_ingest_harness/cases/graphic_elements.py`

Edit (strip graphic-elements assertions and parametrizations; tests must still pass against the OCR-only chart path):
- `nemo_retriever/tests/test_harness_run.py`
- `nemo_retriever/tests/test_ocr_version_selection.py`
- `nemo_retriever/tests/test_multimodal_embed.py`
- `nemo_retriever/tests/test_ingest_plans.py`
- `nemo_retriever/tests/test_operator_flags_and_cpu_actors.py`
- `nemo_retriever/tests/test_actor_operators.py`
- `nemo_retriever/tests/test_pipeline_graph.py`

### 12. Documentation

- `docs/docs/extraction/extraction-charts-infographics.md` — replace graphic-elements paragraphs with: charts go through OCR, text is emitted in OCR word order
- `nemo_retriever/README.md` — drop graphic-elements model row from any model tables; update chart-extraction description
- (Helm/harness READMEs covered above)

## Verification

The change is large but mechanical. Verification:

1. `grep -ri "graphic.element\|graphic_element\|graphic-element\|GraphicElement" --include='*.py' --include='*.yaml' --include='*.yml' --include='*.md' --include='*.toml' --include='*.json'` returns zero hits in tracked files.
2. `pytest nemo_retriever/tests api/api_tests` passes.
3. `python ci/scripts/validate_deployment_configs.py` passes.
4. End-to-end: ingest a PDF containing charts with the new pipeline; confirm `chart` entries appear in metadata with OCR text and no `-`-rearranged structure. (Manual.)

## Risks

- **Hidden references via string keys.** YAML configs and Ray actor name lookups may carry the string `nemotron-graphic-elements-v1` or `chart_graphic_elements` in places grep won't catch (e.g., dynamic dictionary keys built from settings). Mitigation: run the integration test suite + load default_pipeline.yaml and assert build_plan succeeds.
- **Schema regression in `chart` column.** Downstream consumers may expect the previous structured-text format. With graphic-elements removed, `chart[*].text` is plain OCR text. Document this in the PR description; metadata schema (column name + entry shape `{bbox_xyxy_norm, text}`) is unchanged.
- **Helm overrides.** Each per-GPU values file individually overrides graphic-elements resources. Missing one leaves a dangling reference. Mitigation: explicit grep + each file edited.

## Out of scope

- Renaming the `chart` content type or column.
- Changing the OCR model selection or version.
- Modifying table extraction (it has its own dedicated model and pipeline).
- Backwards-compatibility shims for the old `use_graphic_elements` config key — drop it cleanly per project convention.
