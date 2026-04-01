from pathlib import Path

import pandas as pd

from nemo_retriever.ingest_modes.inprocess import GraphOperatorDescriptor
from nemo_retriever.ingest_modes.inprocess import InProcessIngestor
from nemo_retriever.ingest_modes.inprocess import _process_doc_with_descriptors
from nemo_retriever.params import ASRParams
from nemo_retriever.params import AudioChunkParams
from nemo_retriever.params import CaptionParams
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams


def test_inprocess_plan_records_text_pipeline_stages_and_sinks(tmp_path: Path) -> None:
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("hello world", encoding="utf-8")

    ingestor = (
        InProcessIngestor(documents=[])
        .files([str(txt_file)])
        .extract_txt(TextChunkParams(max_tokens=64))
        .split(TextChunkParams(max_tokens=32))
        .caption(CaptionParams(endpoint_url="http://caption.example/v1"))
        .embed(
            EmbedParams(
                model_name="nvidia/llama-nemotron-embed-1b-v2",
                embedding_endpoint="http://embed.example/v1",
            )
        )
        .vdb_upload(VdbUploadParams())
        .save_to_disk(str(tmp_path / "out"))
    )

    assert ingestor._plan.extraction_mode == "text"
    assert ingestor._plan.text_params is not None
    assert ingestor._plan.text_params.max_tokens == 64
    assert ingestor._plan.split_params is not None
    assert ingestor._plan.split_params.max_tokens == 32
    assert ingestor._plan.caption_params is not None
    assert ingestor._plan.embed_params is not None
    assert ingestor._plan.stage_order == ["split", "caption", "embed"]
    assert ingestor._plan.sink_order == ["vdb_upload", "save_to_disk"]

    per_doc_tasks, post_tasks = ingestor.get_pipeline_tasks()

    assert [getattr(func, "__name__", func.__class__.__name__) for func, _ in per_doc_tasks] == [
        "split_df",
        "caption_images",
        "embed_text_main_text_embed",
    ]
    assert [getattr(func, "__name__", func.__class__.__name__) for func, _ in post_tasks] == [
        "upload_embeddings_to_lancedb_inprocess",
        "save_dataframe_to_disk_json",
    ]


def test_inprocess_plan_records_audio_extraction_params() -> None:
    ingestor = InProcessIngestor(documents=[]).extract_audio(
        params=AudioChunkParams(split_type="size", split_interval=12345),
        asr_params=ASRParams(audio_endpoints=("localhost:50051", None)),
    )

    assert ingestor._plan.extraction_mode == "audio"
    assert ingestor._plan.audio_chunk_params is not None
    assert ingestor._plan.audio_chunk_params.split_interval == 12345
    assert ingestor._plan.asr_params is not None
    assert ingestor._plan.asr_params.audio_endpoints == ("localhost:50051", None)

    per_doc_tasks, post_tasks = ingestor.get_pipeline_tasks()
    assert [getattr(func, "__name__", func.__class__.__name__) for func, _ in per_doc_tasks] == ["apply_asr_to_df"]
    assert post_tasks == []


def test_inprocess_ingest_uses_plan_when_tasks_are_cleared(tmp_path: Path) -> None:
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("hello world\nsecond line", encoding="utf-8")

    ingestor = InProcessIngestor(documents=[]).files([str(txt_file)]).extract_txt(TextChunkParams(max_tokens=64))

    results = ingestor.ingest()

    assert len(results) == 1
    assert isinstance(results[0], pd.DataFrame)
    assert not results[0].empty
    assert "text" in results[0].columns


def test_inprocess_sequential_ingest_uses_graph_directly(tmp_path: Path, monkeypatch) -> None:
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("hello world\nsecond line", encoding="utf-8")

    ingestor = InProcessIngestor(documents=[]).files([str(txt_file)]).extract_txt(TextChunkParams(max_tokens=64))

    def _fail_translate(*args, **kwargs):
        raise AssertionError("sequential ingest should not depend on graph-to-task translation")

    monkeypatch.setattr(ingestor, "_translate_graph_node_to_tasks", _fail_translate)

    results = ingestor.ingest()

    assert len(results) == 1
    assert isinstance(results[0], pd.DataFrame)
    assert not results[0].empty


def test_process_doc_with_descriptors_supports_text_root(tmp_path: Path) -> None:
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("hello world\nsecond line", encoding="utf-8")

    root_descriptor = GraphOperatorDescriptor(
        operator_module="nemo_retriever.txt.ray_data",
        operator_qualname="TxtSplitActor",
        operator_kwargs={"params": TextChunkParams(max_tokens=64)},
    )

    result = _process_doc_with_descriptors(str(txt_file), root_descriptor, [])

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "text" in result.columns
