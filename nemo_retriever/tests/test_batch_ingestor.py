from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from nemo_retriever.ingest_modes.batch import BatchIngestor
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams


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
        self.write_parquet_calls: list[str] = []

    def repartition(self, *, target_num_rows_per_block: int):
        self.repartition_calls.append(target_num_rows_per_block)
        return self

    def map_batches(self, fn, **kwargs):
        self.map_batches_calls.append({"fn": fn, **kwargs})
        return self

    def write_parquet(self, path: str):
        self.write_parquet_calls.append(path)


class _LegacySinkRecorder:
    def __init__(self, dataset: _DummyDataset) -> None:
        self.calls: list[tuple[str, object]] = []
        self._dataset = dataset

    def vdb_upload(self, params) -> "_LegacySinkRecorder":
        self.calls.append(("vdb_upload", params))
        return self

    def get_dataset(self) -> _DummyDataset:
        return self._dataset


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
        lambda cluster_resources, allow_no_gpu=False: {"plan": "dummy"},
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
    ingestor._graph_plan.source_dataset = dataset
    ingestor._graph_plan.set_extraction(mode="text", text_params=TextChunkParams())

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

    assert dataset.repartition_calls == []
    assert dataset.map_batches_calls == []
    assert ingestor._graph_plan.stage_order == ["embed"]

    overrides = ingestor._build_graph_node_overrides()
    assert overrides["_BatchEmbedActor"]["batch_size"] == 1
    assert overrides["_BatchEmbedActor"]["num_gpus"] == 1.0
    compute = overrides["_BatchEmbedActor"]["concurrency"]
    assert compute == 1
    assert ingestor._graph_plan.embed_params is not None
    assert ingestor._graph_plan.embed_params.batch_tuning.embed_batch_size == 1


def test_graph_batch_save_intermediate_results_is_plan_driven(tmp_path, monkeypatch) -> None:
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
        lambda _ray: _DummyClusterResources(),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_requested_plan",
        lambda cluster_resources, allow_no_gpu=False: {"plan": "dummy"},
    )

    ingestor = BatchIngestor(documents=[])
    dataset = _DummyDataset()
    ingestor._graph_plan.source_dataset = dataset
    ingestor._graph_plan.set_extraction(mode="text", text_params=TextChunkParams())

    out_dir = tmp_path / "out"
    ingestor.save_intermediate_results(str(out_dir))

    assert dataset.write_parquet_calls == []
    assert ingestor._graph_plan.sink_order == ["save_intermediate_results"]

    ingestor._run_graph_sinks(dataset)

    assert len(dataset.write_parquet_calls) == 1
    assert dataset.write_parquet_calls[0] == str(out_dir.resolve())


def test_graph_batch_vdb_upload_is_recorded_as_sink_and_replayed_after_transforms(tmp_path, monkeypatch) -> None:
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
        lambda _ray: _DummyClusterResources(),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_requested_plan",
        lambda cluster_resources, allow_no_gpu=False: {"plan": "dummy"},
    )

    ingestor = BatchIngestor(documents=[])
    dataset = _DummyDataset()
    ingestor._graph_plan.source_dataset = dataset
    ingestor._graph_plan.set_extraction(mode="text", text_params=TextChunkParams())

    ingestor.embed(EmbedParams(model_name="nvidia/llama-nemotron-embed-vl-1b-v2", embed_granularity="page"))
    vdb_params = VdbUploadParams()
    out_dir = tmp_path / "out"

    ingestor.vdb_upload(vdb_params)
    ingestor.save_intermediate_results(str(out_dir))

    assert ingestor._graph_plan.stage_order == ["embed"]
    assert ingestor._graph_plan.sink_order == ["vdb_upload", "save_intermediate_results"]

    legacy_dataset = _DummyDataset()
    legacy = _LegacySinkRecorder(legacy_dataset)
    ingestor._run_legacy_sinks(legacy)

    assert legacy.calls == [("vdb_upload", vdb_params)]
    assert legacy_dataset.write_parquet_calls == [str(out_dir.resolve())]
