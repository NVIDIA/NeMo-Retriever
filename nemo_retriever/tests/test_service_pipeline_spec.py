# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 unit tests for the per-request PipelineSpec wire format.

Covers three layers:

* the client-side ``ServiceIngestor`` fluent builders translate to the
  right ``_pipeline_spec`` shape;
* the server-side ``validate_pipeline_spec`` policy accepts well-formed
  specs and rejects trust-sensitive overrides; and
* the worker-side merge preserves server-owned keys regardless of the
  client spec.
"""

from __future__ import annotations

import pytest

from nemo_retriever.params import DedupParams, EmbedParams, ExtractParams
from nemo_retriever.service.config import PipelineOverridesConfig
from nemo_retriever.service.models.pipeline_spec import PipelineSpec
from nemo_retriever.service.policy import PolicyError, validate_pipeline_spec
from nemo_retriever.service.services.pipeline_executor import (
    _build_graph_ingestor_from_spec,
    _merge_server_owned,
    _TRUST_OWNED_EMBED_KEYS,
    _TRUST_OWNED_EXTRACT_KEYS,
)
from nemo_retriever.service_ingestor import ServiceIngestor


# ----------------------------------------------------------------------
# Client side: fluent → spec dict
# ----------------------------------------------------------------------


def test_serviceingestor_empty_spec_is_none() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    assert ing._pipeline_payload() is None


def test_extract_records_stage_and_params() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract(ExtractParams(extract_text=False, dpi=300))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "pdf"
    assert payload["stage_order"] == ["extract"]
    assert payload["extract_params"]["extract_text"] is False
    assert payload["extract_params"]["dpi"] == 300
    assert "page_elements_invoke_url" not in payload["extract_params"]
    assert "api_key" not in payload["extract_params"]


def test_extract_image_files_sets_image_mode() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract_image_files()
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "image"


def test_dedup_and_embed_add_stage_order() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract().dedup(DedupParams(iou_threshold=0.7)).embed(EmbedParams(inference_batch_size=64))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["stage_order"] == ["extract", "dedup", "embed"]
    assert payload["dedup_params"]["iou_threshold"] == 0.7
    assert payload["embed_params"]["inference_batch_size"] == 64


def test_pdf_split_config_round_trips_via_spec() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.pdf_split_config(pages_per_chunk=16)
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["pdf_split"]["pages_per_chunk"] == 16


def test_split_method_records_split_config() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.split({"pdf": {"max_tokens": 512, "overlap_tokens": 32}})
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["split_config"] == {"pdf": {"max_tokens": 512, "overlap_tokens": 32}}


def test_all_tasks_seeds_canonical_stage_order() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.all_tasks()
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["stage_order"] == ["extract", "dedup", "embed"]


def test_client_rejects_server_owned_keys() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="server-owned"):
        ing.extract(ExtractParams(page_elements_invoke_url="http://attacker/"))


def test_future_phase_methods_raise_informative_error() -> None:
    """Methods deferred to follow-up phases still produce a clear error.

    ``store`` / ``webhook`` / ``vdb_upload`` (sinks) moved out in Phase 2,
    ``save_to_disk`` in Phase 3, ``caption`` in Phase 4, ``vdb_upload``
    sidecar metadata in Phase 6. ``udf`` is the only remaining stub.
    """
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(NotImplementedError, match="Phase 5"):
        ing.udf("noop")


# ----------------------------------------------------------------------
# Policy: accept / reject
# ----------------------------------------------------------------------


def test_validate_returns_none_for_empty_spec() -> None:
    policy = PipelineOverridesConfig().to_policy()
    assert validate_pipeline_spec(None, policy) is None
    assert validate_pipeline_spec(PipelineSpec(), policy) is None


def test_validate_accepts_default_allowlist() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(
        extract_params={"extract_text": False, "dpi": 300},
        embed_params={"inference_batch_size": 64},
        dedup_params={"iou_threshold": 0.5},
        stage_order=["extract", "dedup", "embed"],
    )
    out = validate_pipeline_spec(spec, policy)
    assert out is spec  # returned by reference when unchanged


def test_validate_rejects_endpoint_url() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(extract_params={"page_elements_invoke_url": "http://attacker/"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403
    assert "trust-sensitive" in exc.value.detail


def test_validate_rejects_api_key() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(embed_params={"api_key": "leaked-token"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403


def test_validate_rejects_unallowed_key_in_allow_list_mode() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(extract_params={"not_a_real_field": True})
    with pytest.raises(PolicyError):
        validate_pipeline_spec(spec, policy)


def test_validate_allows_extra_key_when_operator_widens() -> None:
    cfg = PipelineOverridesConfig(extra_extract_keys=["weird_dev_flag"])
    spec = PipelineSpec(extract_params={"weird_dev_flag": True, "dpi": 300})
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


def test_validate_reject_mode_blocks_any_override() -> None:
    cfg = PipelineOverridesConfig(mode="reject")
    spec = PipelineSpec(extract_params={"dpi": 300})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, cfg.to_policy())
    assert exc.value.status_code == 403


def test_validate_allow_all_mode_still_blocks_endpoints() -> None:
    cfg = PipelineOverridesConfig(mode="allow_all")
    policy = cfg.to_policy()
    # "shape" keys pass freely:
    spec = PipelineSpec(extract_params={"any_dev_only_flag": True})
    assert validate_pipeline_spec(spec, policy) is spec
    # but the denylist still bites:
    spec2 = PipelineSpec(extract_params={"ocr_invoke_url": "http://x/"})
    with pytest.raises(PolicyError):
        validate_pipeline_spec(spec2, policy)


def test_validate_rejects_caption_without_endpoint() -> None:
    """Without an operator-configured caption endpoint, the stage is forbidden."""
    cfg = PipelineOverridesConfig()
    policy = cfg.to_policy(caption_enabled=False)
    spec = PipelineSpec(caption_params={"prompt": "Describe"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403


# ----------------------------------------------------------------------
# Worker merge: server-owned keys always win
# ----------------------------------------------------------------------


def test_merge_preserves_server_extract_endpoints() -> None:
    base = {
        "page_elements_invoke_url": "http://server/page_elements",
        "ocr_invoke_url": "http://server/ocr",
        "api_key": "server-token",
        "dpi": 150,
    }
    override = {"dpi": 600, "page_elements_invoke_url": "http://attacker/"}
    merged = _merge_server_owned(base, override, _TRUST_OWNED_EXTRACT_KEYS)
    assert merged["dpi"] == 600
    assert merged["page_elements_invoke_url"] == "http://server/page_elements"
    assert merged["ocr_invoke_url"] == "http://server/ocr"
    assert merged["api_key"] == "server-token"


def test_merge_preserves_server_embed_endpoints() -> None:
    base = {"embed_invoke_url": "http://server/embed", "api_key": "k"}
    override = {"embed_invoke_url": "http://attacker/", "inference_batch_size": 8}
    merged = _merge_server_owned(base, override, _TRUST_OWNED_EMBED_KEYS)
    assert merged["embed_invoke_url"] == "http://server/embed"
    assert merged["api_key"] == "k"
    assert merged["inference_batch_size"] == 8


def test_build_graph_ingestor_applies_spec_extraction_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``extraction_mode='image'`` calls extract_image_files on the GraphIngestor."""
    base_extract = {"page_elements_invoke_url": "http://server/page_elements"}
    spec = {"extraction_mode": "image", "extract_params": {"dpi": 300}, "stage_order": ["extract"]}

    ingestor, mode, has_vdb = _build_graph_ingestor_from_spec(
        "stub.png",
        b"\x89PNG\r\n",
        base_extract,
        None,
        spec,
    )
    assert mode == "image"
    assert has_vdb is False
    assert ingestor._extraction_mode == "image"
    assert ingestor._extract_params is not None
    assert ingestor._extract_params.dpi == 300
    assert ingestor._extract_params.page_elements_invoke_url == "http://server/page_elements"
