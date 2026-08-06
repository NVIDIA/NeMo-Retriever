# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_retriever.models import hf_model_registry as registry


def test_extraction_hf_repos_have_pinned_revisions():
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-ocr-v1"] == "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb"
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-ocr-v2"] == "86cacb0467fa4f7ce54342fdb250825e0d928ae7"
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-page-elements-v3"] == "df62dbb631502575ac4d43b44d700b1674ab1d56"
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/nemotron-table-structure-v1"] == "9350162faa1110320af62699105780b0c87b73ad"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"]
        == "5d250e2e111dc5e1434131bdf3d590c27a878ade"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"]
        == "7394488badb786e1decc0e00e308de1cab9560e6"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"]
        == "b8d3c170d9ee3a078917ef9bfd508eff988d6de7"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"]
        == "24e67ea000b7c2837fc8f9488aa2008524fac8ba"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"]
        == "6647b845a4b786c6e2c7adb1b6a909e1aa71fac2"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"]
        == "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
    )


@pytest.fixture
def local_checkpoint(tmp_path):
    """A valid on-disk model directory, i.e. one containing ``config.json``."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_local_dir_requires_explicit_opt_in_to_bypass_revision_pin(local_checkpoint):
    with pytest.raises(ValueError, match="No pinned HuggingFace revision"):
        registry.get_hf_revision(str(local_checkpoint))

    assert registry.get_hf_revision(str(local_checkpoint), allow_local_path=True) is None


def test_local_dir_without_model_config_does_not_bypass_revision_pin(tmp_path):
    with pytest.raises(ValueError, match="has no config.json"):
        registry.get_hf_revision(str(tmp_path), allow_local_path=True)


def test_registered_hub_id_still_pinned():
    assert registry.get_hf_revision("nvidia/llama-nemotron-embed-1b-v2") == "b4caa8456edd360b3b4e938d94ed4398dd437fad"


def test_unregistered_hub_id_still_raises():
    with pytest.raises(ValueError, match="No pinned HuggingFace revision"):
        registry.get_hf_revision("some-org/not-registered")


def test_unregistered_hub_id_non_strict_returns_none():
    assert registry.get_hf_revision("some-org/not-registered", strict=False) is None


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
    class LocalEntryNotFoundError(Exception):
        pass

    LocalEntryNotFoundError.__module__ = "huggingface_hub.errors"

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
