import pytest

from tests import _have_ffmpeg_binary
from nemo_retriever.graph.ingestor_runtime import batch_tuning_to_node_overrides
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.graph.ingestor_runtime import build_inprocess_graph
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.ocr.ocr import OCRActor
from nemo_retriever.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.text_embed.operators import _BatchEmbedActor
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.ingest_plans import BaseIngestPlan
from nemo_retriever.params import ASRParams
from nemo_retriever.params import AudioChunkParams
from nemo_retriever.params import BatchTuningParams
from nemo_retriever.params import CaptionParams
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import StoreParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.params import WebhookParams
from nemo_retriever.utils.ray_resource_hueristics import ClusterResources
from nemo_retriever.utils.ray_resource_hueristics import Resources


def _linear_nodes(graph):
    node = graph.roots[0]
    nodes = []
    while True:
        nodes.append(node)
        if not node.children:
            return nodes
        node = node.children[0]


def test_base_ingest_plan_carries_split_config() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="pdf", extract_params=ExtractParams())
    plan.split_config = {"pdf": TextChunkParams(max_tokens=128)}
    plan.caption_params = CaptionParams(endpoint_url="http://caption.example/v1")
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )

    plan.record_stage("caption")
    plan.record_stage("embed")

    execution_plan = plan.build_execution_plan()

    assert execution_plan.extraction_mode == "pdf"
    assert execution_plan.split_config["pdf"].max_tokens == 128
    assert [stage.name for stage in execution_plan.stages] == ["caption", "embed"]


def test_base_ingest_plan_builds_audio_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="audio", audio_chunk_params=AudioChunkParams(split_type="size", split_interval=42))

    execution_plan = plan.build_execution_plan()

    assert execution_plan.extraction_mode == "audio"
    assert execution_plan.audio_chunk_params is not None
    assert execution_plan.audio_chunk_params.split_interval == 42
    assert execution_plan.has_extraction() is True


def test_build_graph_accepts_execution_plan_with_split_config() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="text", text_params=TextChunkParams(max_tokens=64))
    plan.split_config = {"text": TextChunkParams(max_tokens=32)}
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )
    plan.record_stage("embed")

    graph = build_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    # Text path uses MultiTypeExtractOperator; split_config['text'] = 32-token
    # params is forwarded into TxtSplitActor inside the operator (no separate
    # TextChunkActor at graph level here).
    assert names == ["MultiTypeExtractOperator", "_BatchEmbedActor"]


def test_build_graph_inserts_ingest_vdb_before_webhook() -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
        ),
        vdb_upload_params=VdbUploadParams(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp/t"}),
        webhook_params=WebhookParams(endpoint_url="http://webhook.example/hook"),
        stage_order=(),
    )

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names[-2] == "IngestVdbOperator"
    assert names[-1] == "WebhookNotifyOperator"


def test_build_graph_vdb_from_execution_plan_sink() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="text", text_params=TextChunkParams(max_tokens=64))
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embed_invoke_url="http://embed.example/v1",
    )
    plan.vdb_upload_params = VdbUploadParams(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp/p"})
    plan.record_stage("embed")
    plan.record_sink("vdb_upload")

    graph = build_graph(execution_plan=plan.build_execution_plan())
    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert "IngestVdbOperator" in names
    assert names.index("IngestVdbOperator") > names.index("_BatchEmbedActor")


@pytest.mark.parametrize(
    "ocr_version, expected_actor_class, expected_actor_name",
    [
        ("v2", OCRActor, "OCRActor"),
        ("v1", OCRActor, "OCRActor"),
    ],
)
def test_build_graph_keeps_archetype_operator_classes(
    ocr_version: str, expected_actor_class: type, expected_actor_name: str
) -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            ocr_version=ocr_version,
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2", embed_invoke_url="http://embed.example/v1"
        ),
    )

    nodes = _linear_nodes(graph)

    assert [node.name for node in nodes] == [
        "DocToPdfConversionActor",
        "PDFSplitActor",
        "PDFExtractionActor",
        "PageElementDetectionActor",
        expected_actor_name,
        "UDFOperator",
        "_BatchEmbedActor",
    ]
    assert nodes[3].operator_class is PageElementDetectionActor
    assert nodes[4].operator_class is expected_actor_class
    assert nodes[4].operator_kwargs["ocr_version"] == ocr_version
    assert nodes[-1].operator_class is _BatchEmbedActor
    assert issubclass(nodes[3].operator_class, ArchetypeOperator)
    assert issubclass(nodes[4].operator_class, ArchetypeOperator)
    assert issubclass(nodes[-1].operator_class, ArchetypeOperator)


@pytest.mark.parametrize(
    "ocr_version, expected_node_name, expected_cpu_class_name",
    [
        ("v2", "OCRActor", "OCRCPUActor"),
        ("v1", "OCRActor", "OCRCPUActor"),
    ],
)
def test_build_graph_resolves_endpoint_configured_nodes_to_cpu_variants(
    ocr_version: str, expected_node_name: str, expected_cpu_class_name: str
) -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=False,
            page_elements_invoke_url="http://page.example/v1",
            ocr_invoke_url="http://ocr.example/v1",
            table_structure_invoke_url="http://table.example/v1",
            graphic_elements_invoke_url="http://graphic.example/v1",
            ocr_version=ocr_version,
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2", embed_invoke_url="http://embed.example/v1"
        ),
    )

    resolved = graph.resolve(Resources(cpu_count=8, gpu_count=4))
    classes = {node.name: node.operator_class for node in _linear_nodes(resolved)}

    assert classes["PageElementDetectionActor"].__name__ == "PageElementDetectionCPUActor"
    assert classes["TableStructureActor"].__name__ == "TableStructureCPUActor"
    assert classes["GraphicElementsActor"].__name__ == "GraphicElementsCPUActor"
    assert classes[expected_node_name].__name__ == expected_cpu_class_name
    assert classes["_BatchEmbedActor"].__name__ == "_BatchEmbedCPUActor"
    assert issubclass(classes["PageElementDetectionActor"], CPUOperator)
    assert issubclass(classes[expected_node_name], CPUOperator)
    assert issubclass(classes["_BatchEmbedActor"], CPUOperator)


@pytest.mark.parametrize(
    "ocr_version, expected_node_name, expected_archetype_class",
    [
        ("v2", "OCRActor", OCRActor),
        ("v1", "OCRActor", OCRActor),
    ],
)
def test_build_graph_resolves_local_nodes_to_gpu_variants_when_gpus_available(
    ocr_version: str, expected_node_name: str, expected_archetype_class: type
) -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            ocr_version=ocr_version,
        ),
        embed_params=EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"),
    )

    resolved = graph.resolve(Resources(cpu_count=8, gpu_count=1))
    classes = {node.name: node.operator_class for node in _linear_nodes(resolved)}

    assert classes["PageElementDetectionActor"] is not PageElementDetectionActor
    assert classes[expected_node_name] is not expected_archetype_class
    assert classes["_BatchEmbedActor"] is not _BatchEmbedActor
    assert issubclass(classes["PageElementDetectionActor"], GPUOperator)
    assert issubclass(classes[expected_node_name], GPUOperator)
    assert issubclass(classes["_BatchEmbedActor"], GPUOperator)


@pytest.mark.parametrize(
    "ocr_version, expected_actor_name",
    [
        ("v2", "OCRActor"),
        ("v1", "OCRActor"),
    ],
)
def test_batch_tuning_to_node_overrides_auto_cpu_only_when_no_gpus(ocr_version: str, expected_actor_name: str) -> None:
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=16, gpu_count=0),
        available_resources=Resources(cpu_count=16, gpu_count=0),
    )
    extract_params = ExtractParams(
        method="ocr",
        ocr_version=ocr_version,
        batch_tuning=BatchTuningParams(
            gpu_page_elements=0.5,
            gpu_ocr=0.5,
            gpu_nemotron_parse=1.0,
            page_elements_workers=3,
            ocr_workers=4,
            nemotron_parse_workers=2,
        ),
    )
    embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        batch_tuning=BatchTuningParams(
            gpu_embed=0.5,
            embed_workers=5,
        ),
    )

    overrides = batch_tuning_to_node_overrides(
        extract_params=extract_params,
        embed_params=embed_params,
        cluster_resources=cluster,
    )

    assert overrides["_BatchEmbedActor"]["num_gpus"] == 0.0
    assert overrides[expected_actor_name]["num_gpus"] == 0.0
    assert overrides["PageElementDetectionActor"]["num_gpus"] == 0.0
    assert overrides["NemotronParseActor"]["num_gpus"] == 0.0
    assert overrides["_BatchEmbedActor"]["concurrency"] == 5
    assert overrides[expected_actor_name]["concurrency"] == 4
    assert overrides["PageElementDetectionActor"]["concurrency"] == 3
    assert overrides["NemotronParseActor"]["concurrency"] == 2


def test_batch_tuning_to_node_overrides_adds_default_store_tuning() -> None:
    overrides = batch_tuning_to_node_overrides(
        extract_params=None,
        embed_params=None,
        store_params=StoreParams(storage_uri="memory://stored"),
    )

    assert overrides["StoreOperator"] == {"concurrency": (1, 4, 1), "num_cpus": 0.1}


def test_batch_tuning_to_node_overrides_honors_store_tuning() -> None:
    store_params = StoreParams(
        storage_uri="memory://stored",
        batch_tuning=BatchTuningParams(store_workers=1),
    )

    overrides = batch_tuning_to_node_overrides(
        extract_params=None,
        embed_params=None,
        store_params=store_params,
    )

    assert overrides["StoreOperator"] == {"concurrency": 1, "num_cpus": 0.1}


def test_graph_ingestor_autodetects_no_gpu_for_batch_overrides(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeRay:
        @staticmethod
        def is_initialized():
            return True

    class _FakeExecutor:
        def __init__(self, graph, **kwargs):
            self.graph = graph

        def ingest(self, data):
            return {"data": data, "graph": self.graph}

    def _fake_batch_tuning_to_node_overrides(
        extract_params, embed_params, cluster_resources=None, allow_no_gpu=None, **kwargs
    ):
        captured["allow_no_gpu"] = allow_no_gpu
        captured["cluster_resources"] = cluster_resources
        return {}

    cluster = ClusterResources(
        total_resources=Resources(cpu_count=16, gpu_count=0),
        available_resources=Resources(cpu_count=16, gpu_count=0),
    )

    monkeypatch.setattr("nemo_retriever.graph_ingestor.build_graph", lambda **kwargs: Graph())
    monkeypatch.setattr(
        "nemo_retriever.graph_ingestor.batch_tuning_to_node_overrides", _fake_batch_tuning_to_node_overrides
    )
    monkeypatch.setattr("nemo_retriever.graph_ingestor.gather_cluster_resources", lambda ray: cluster)
    monkeypatch.setattr("nemo_retriever.graph_ingestor.RayDataExecutor", _FakeExecutor)
    monkeypatch.setattr("ray.is_initialized", _FakeRay.is_initialized)

    ingestor = GraphIngestor(run_mode="batch", documents=["/tmp/input.pdf"])
    ingestor.extract(ExtractParams(method="ocr"))
    ingestor.embed(EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"))

    result = ingestor.ingest()

    assert captured["allow_no_gpu"] is True
    assert captured["cluster_resources"] == cluster
    assert result["data"] == ["/tmp/input.pdf"]


def test_build_inprocess_graph_supports_text_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="text", text_params=TextChunkParams(max_tokens=64))
    plan.split_config = {"text": TextChunkParams(max_tokens=32)}
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )
    plan.record_stage("embed")

    graph = build_inprocess_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["MultiTypeExtractOperator", "_BatchEmbedActor"]


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_build_inprocess_graph_supports_audio_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(
        mode="audio",
        audio_chunk_params=AudioChunkParams(split_type="size", split_interval=42),
        asr_params=ASRParams(audio_endpoints=("localhost:50051", None)),
    )

    graph = build_inprocess_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["MediaChunkActor", "ASRActor"]


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_build_graph_uses_explicit_audio_graph_for_audio_extract_method() -> None:
    graph = build_graph(
        extract_params=ExtractParams(method="audio"),
        audio_chunk_params=AudioChunkParams(split_type="size", split_interval=42),
    )

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["MediaChunkActor", "ASRActor"]
