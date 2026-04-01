from nemo_retriever.graph.ingestor_runtime import build_batch_graph
from nemo_retriever.graph.ingestor_runtime import build_inprocess_graph
from nemo_retriever.ingest_modes._plans import BaseIngestPlan
from nemo_retriever.params import ASRParams
from nemo_retriever.params import AudioChunkParams
from nemo_retriever.params import CaptionParams
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams


def test_base_ingest_plan_builds_ordered_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="pdf", extract_params=ExtractParams())
    plan.split_params = TextChunkParams(max_tokens=128)
    plan.caption_params = CaptionParams(endpoint_url="http://caption.example/v1")
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )
    plan.vdb_upload_params = VdbUploadParams()

    plan.record_stage("caption")
    plan.record_stage("split")
    plan.record_stage("embed")
    plan.record_stage("caption")
    plan.record_sink("vdb_upload")

    execution_plan = plan.build_execution_plan()

    assert execution_plan.extraction_mode == "pdf"
    assert execution_plan.extract_params is not None
    assert [stage.name for stage in execution_plan.stages] == ["split", "embed", "caption"]
    assert execution_plan.stages[0].params.max_tokens == 128
    assert execution_plan.stages[1].params.model_name == "nvidia/llama-nemotron-embed-1b-v2"
    assert execution_plan.stages[2].params.endpoint_url == "http://caption.example/v1"
    assert [sink.name for sink in execution_plan.sinks] == ["vdb_upload"]


def test_base_ingest_plan_builds_audio_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="audio", audio_chunk_params=AudioChunkParams(split_type="size", split_interval=42))

    execution_plan = plan.build_execution_plan()

    assert execution_plan.extraction_mode == "audio"
    assert execution_plan.audio_chunk_params is not None
    assert execution_plan.audio_chunk_params.split_interval == 42
    assert execution_plan.has_extraction() is True


def test_build_batch_graph_accepts_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="text", text_params=TextChunkParams(max_tokens=64))
    plan.split_params = TextChunkParams(max_tokens=32)
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )
    plan.record_stage("split")
    plan.record_stage("embed")

    graph = build_batch_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["MultiTypeExtractOperator", "TextChunkActor", "_BatchEmbedActor"]


def test_build_inprocess_graph_accepts_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="pdf", extract_params=ExtractParams(extract_text=True))
    plan.split_params = TextChunkParams(max_tokens=32)
    plan.caption_params = CaptionParams(endpoint_url="http://caption.example/v1")
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )
    plan.record_stage("split")
    plan.record_stage("caption")
    plan.record_stage("embed")

    graph = build_inprocess_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names[-4:] == ["TextChunkActor", "CaptionActor", "UDFOperator", "_BatchEmbedActor"]


def test_build_inprocess_graph_supports_text_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="text", text_params=TextChunkParams(max_tokens=64))
    plan.split_params = TextChunkParams(max_tokens=32)
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )
    plan.record_stage("split")
    plan.record_stage("embed")

    graph = build_inprocess_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["TxtSplitActor", "TextChunkActor", "_BatchEmbedActor"]


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
