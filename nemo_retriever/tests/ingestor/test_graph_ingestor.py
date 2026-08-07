from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor


def test_graph_ingestor_applies_parallel_cpu_overrides_for_batch_text(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeExecutor:
        def __init__(self, graph, **kwargs):
            self.graph = graph
            captured["node_overrides"] = kwargs["node_overrides"]

        def ingest(self, data):
            return {"data": data, "graph": self.graph}

    cluster = ClusterResources(
        total_resources=Resources(cpu_count=32, gpu_count=1),
        available_resources=Resources(cpu_count=32, gpu_count=1),
    )

    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.build_graph", lambda **kwargs: Graph())
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.GraphIngestor._ensure_batch_runtime",
        lambda self: (object(), cluster),
    )
    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.RayDataExecutor", _FakeExecutor)

    ingestor = GraphIngestor(run_mode="batch", documents=["/tmp/input.txt"])
    ingestor.extract(extraction_mode="text")
    ingestor.embed(EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"))
    ingestor.ingest()

    assert captured["node_overrides"]["MultiTypeExtractOperator"] == {
        "concurrency": 8,
        "num_cpus": 1,
        "num_gpus": 0.0,
    }
