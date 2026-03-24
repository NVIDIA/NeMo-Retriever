from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from nemo_retriever.ingest_modes.batch import BatchIngestor
from nemo_retriever.params import EmbedParams


class _DummyClusterResources:
    def total_cpu_count(self) -> int:
        return 4

    def total_gpu_count(self) -> int:
        return 0

    def available_cpu_count(self) -> int:
        return 4

    def available_gpu_count(self) -> int:
        return 0


class _DummyGpuClusterResources:
    def total_cpu_count(self) -> int:
        return 16

    def total_gpu_count(self) -> int:
        return 2

    def available_cpu_count(self) -> int:
        return 16

    def available_gpu_count(self) -> int:
        return 2


class _DummyDataset:
    def __init__(self) -> None:
        self.repartition_calls: list[int] = []
        self.map_batches_calls: list[dict[str, object]] = []

    def repartition(self, *, target_num_rows_per_block: int):
        self.repartition_calls.append(target_num_rows_per_block)
        return self

    def map_batches(self, fn, **kwargs):
        self.map_batches_calls.append({"fn": fn, **kwargs})
        return self


def test_batch_ingestor_filters_none_runtime_env_vars(monkeypatch) -> None:
    captured: dict[str, object] = {}
    dummy_ctx = SimpleNamespace(enable_rich_progress_bars=False, use_ray_tqdm=True)

    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_hf_cache_dir",
        lambda: "/tmp/hf-cache",
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.ray.init",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.rd.DataContext.get_current",
        lambda: dummy_ctx,
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.gather_cluster_resources",
        lambda _ray: _DummyClusterResources(),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_requested_plan",
        lambda cluster_resources: {"plan": "dummy"},
    )

    BatchIngestor(documents=[])

    assert captured["runtime_env"] == {
        "env_vars": {
            "LOG_LEVEL": "INFO",
            "NEMO_RETRIEVER_HF_CACHE_DIR": "/tmp/hf-cache",
        }
    }
    assert dummy_ctx.enable_rich_progress_bars is True
    assert dummy_ctx.use_ray_tqdm is False


def test_batch_ingestor_embed_honors_batch_tuning(monkeypatch) -> None:
    dummy_ctx = SimpleNamespace(enable_rich_progress_bars=False, use_ray_tqdm=True)

    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.ray.init",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.rd.DataContext.get_current",
        lambda: dummy_ctx,
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.gather_cluster_resources",
        lambda _ray: _DummyGpuClusterResources(),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.rd.ActorPoolStrategy",
        lambda *, initial_size, min_size, max_size: SimpleNamespace(
            initial_size=initial_size,
            min_size=min_size,
            max_size=max_size,
        ),
    )

    ingestor = BatchIngestor(documents=[])
    dataset = _DummyDataset()
    ingestor._rd_dataset = dataset

    ingestor.embed(
        EmbedParams(
            model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
            embed_granularity="page",
            batch_tuning={
                "embed_workers": 1,
                "embed_batch_size": 1,
                "gpu_embed": 1.0,
            },
        )
    )

    assert dataset.repartition_calls == [1]
    assert dataset.map_batches_calls[0]["batch_size"] == 1
    assert dataset.map_batches_calls[1]["batch_size"] == 1
    assert dataset.map_batches_calls[1]["num_gpus"] == 1.0
    compute = dataset.map_batches_calls[1]["compute"]
    assert compute.initial_size == 1
    assert compute.min_size == 1
    assert compute.max_size == 1
