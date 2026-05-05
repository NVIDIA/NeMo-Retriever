# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_retriever.utils import hf_model_registry as registry


def test_extraction_hf_repos_have_pinned_revisions():
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-ocr-v1"] == "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb"
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-ocr-v2"] == "86cacb0467fa4f7ce54342fdb250825e0d928ae7"
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-page-elements-v3"] == "df62dbb631502575ac4d43b44d700b1674ab1d56"
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/nemotron-table-structure-v1"] == "9350162faa1110320af62699105780b0c87b73ad"
    )


def test_hf_hub_download_with_pinned_revision_injects_known_revision(monkeypatch):
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return "/cache/model.bin"

    monkeypatch.setattr(registry, "hf_hub_download", fake_download)

    out = registry.hf_hub_download_with_pinned_revision(
        repo_id="nvidia/nemotron-ocr-v1",
        filename="checkpoints/det_model.pt",
    )

    assert out == "/cache/model.bin"
    assert calls == [
        (
            (),
            {
                "repo_id": "nvidia/nemotron-ocr-v1",
                "filename": "checkpoints/det_model.pt",
                "revision": "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb",
            },
        )
    ]


def test_hf_hub_download_with_pinned_revision_preserves_explicit_revision(monkeypatch):
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return "/cache/model.bin"

    monkeypatch.setattr(registry, "hf_hub_download", fake_download)

    registry.hf_hub_download_with_pinned_revision(
        "nvidia/nemotron-ocr-v1",
        filename="checkpoints/det_model.pt",
        revision="custom-sha",
    )

    assert calls[0][0] == ("nvidia/nemotron-ocr-v1",)
    assert calls[0][1]["revision"] == "custom-sha"


def test_hf_hub_download_with_pinned_revision_adds_startup_context(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def fake_download(*args, **kwargs):
        raise LocalEntryNotFoundError("cache miss")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(registry, "hf_hub_download", fake_download)

    with pytest.raises(RuntimeError) as exc_info:
        registry.hf_hub_download_with_pinned_revision(
            repo_id="nvidia/nemotron-ocr-v1",
            filename="checkpoints/det_model.pt",
        )

    message = str(exc_info.value)
    assert "nvidia/nemotron-ocr-v1" in message
    assert "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb" in message
    assert "checkpoints/det_model.pt" in message
    assert "HF_HUB_OFFLINE=0" in message
    assert "HF token is unset" in message


def test_install_pinned_hf_hub_download_warns_when_module_lacks_downloader(caplog):
    module = SimpleNamespace(__name__="upstream_without_downloader")

    registry.install_pinned_hf_hub_download(module)

    assert "revision pinning was NOT applied" in caplog.text
    assert "upstream_without_downloader" in caplog.text
