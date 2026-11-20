"""RGB arena capacity validation and driver-to-producer configuration."""

import pytest

from nemo_retriever.operators.extract.pdf.gpu_transport import RendererPool, _rgb_pool_capacity


@pytest.mark.parametrize("raw,expected", [(None, 0), ("0", 0), ("3145728", 3145728), ("68719476736", 68719476736)])
def test_rgb_pool_capacity(raw, expected, monkeypatch):
    if raw is None:
        monkeypatch.delenv("CUDA_PDF_RGB_POOL_BYTES", raising=False)
    else:
        monkeypatch.setenv("CUDA_PDF_RGB_POOL_BYTES", raw)
    assert _rgb_pool_capacity() == expected


@pytest.mark.parametrize("raw", ["-1", "1", "1024", "invalid", "1.5", ""])
def test_invalid_rgb_pool_capacity_is_rejected(raw, monkeypatch):
    monkeypatch.setenv("CUDA_PDF_RGB_POOL_BYTES", raw)
    with pytest.raises(ValueError, match="CUDA_PDF_RGB_POOL_BYTES"):
        _rgb_pool_capacity()


def test_pool_capacity_is_forwarded_to_producer_environment(monkeypatch):
    import ray

    options = []
    sentinel = object()

    class Remote:
        def options(self, **kwargs):
            options.append(kwargs)
            return self

        def remote(self, *args):
            return sentinel

    monkeypatch.setattr(ray, "remote", lambda **kwargs: lambda cls: Remote())
    pool = RendererPool("library", rgb_pool_bytes=3145728)
    assert pool.get("1" * 56, "GPU-id") is sentinel
    assert pool.get("1" * 56, "GPU-id") is sentinel
    assert len(options) == 1
    env = options[0]["runtime_env"]["env_vars"]
    assert env["CUDA_PDF_RGB_POOL_BYTES"] == "3145728"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-id"


def test_receiver_capacity_uses_independent_validated_setting(monkeypatch):
    monkeypatch.setenv("CUDA_PDF_RGB_POOL_BYTES", "68719476736")
    monkeypatch.setenv("CUDA_PDF_RGB_RECEIVER_POOL_BYTES", "268435456")
    assert _rgb_pool_capacity("CUDA_PDF_RGB_RECEIVER_POOL_BYTES") == 268435456
    monkeypatch.setenv("CUDA_PDF_RGB_RECEIVER_POOL_BYTES", "1")
    with pytest.raises(ValueError, match="CUDA_PDF_RGB_RECEIVER_POOL_BYTES"):
        _rgb_pool_capacity("CUDA_PDF_RGB_RECEIVER_POOL_BYTES")
