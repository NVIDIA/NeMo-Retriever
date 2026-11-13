# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version isolation, table formatting, and pinned helper loading without inference."""

import base64
from copy import deepcopy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import pytest

from nemo_retriever.common.modality.parse import nemotron_parse35_postprocessing as adapter
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL, NEMOTRON_PARSE_V12_MODEL
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.operators.extract.parse.nemotron_parse import (
    NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
    NEMOTRON_PARSE35_DEFAULT_TASK_PROMPT,
    NemotronParseGPUActor,
    nemotron_parse_pages,
)

TABLE = '<table><tr><th rowspan="1" colspan="2">Group</th></tr><tr><td rowspan="2" colspan="1">A</td></tr></table>'
CHART = '<table><tr><td colspan="1">Chart label</td></tr></table>'
RAW_TABLE = r"\begin{tabular}{cc} A & B \\ 1 & 2 \\ \end{tabular}"
RAW = "<x_0.1><y_0.2>" + RAW_TABLE + "<x_0.9><y_0.8><class_Table>"


@pytest.fixture
def helpers(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    blocks = [
        {"class": "Table", "bbox": [0.1, 0.2, 0.9, 0.8], "text": RAW_TABLE, "content": TABLE},
        {"class": "Chart", "bbox": [0, 0, 1, 1], "text": "chart raw", "content": CHART},
        {"class": "Picture", "bbox": [0, 0, 1, 1], "text": "Label", "content": "Label"},
    ]
    document = Mock(side_effect=lambda _: {"blocks": deepcopy(blocks)})
    convert = Mock(side_effect=lambda text, cls, *, table_format: RAW_TABLE if table_format == "latex" else "| A | B |")
    result = SimpleNamespace(postprocess_document=document, postprocessing=SimpleNamespace(postprocess_text=convert))
    monkeypatch.setattr(adapter, "_load_parse35_helpers", lambda: result)
    return result


@pytest.mark.parametrize("table_format", ["HTML", "simple_html", "markdown", "latex"])
def test_only_tables_change_format(helpers: SimpleNamespace, table_format: str) -> None:
    blocks = adapter._postprocess_parse35_blocks(RAW, table_format=table_format)
    assert blocks[1]["content"] == CHART
    assert blocks[2]["content"] == "Label"
    assert blocks[0]["bbox"] == [0.1, 0.2, 0.9, 0.8]
    assert blocks[0]["text"] == RAW_TABLE
    if table_format == "HTML":
        assert blocks[0]["content"] == TABLE
    elif table_format == "simple_html":
        soup = BeautifulSoup(blocks[0]["content"], "html.parser")
        assert soup.th.attrs == {"colspan": "2"}
        assert soup.td.attrs == {"rowspan": "2"}
    else:
        assert blocks[0]["content"] == (RAW_TABLE if table_format == "latex" else "| A | B |")
        helpers.postprocessing.postprocess_text.assert_called_once_with(RAW_TABLE, "Table", table_format=table_format)


def test_markdown_conversion_failure_preserves_html(helpers: SimpleNamespace, caplog: pytest.LogCaptureFixture) -> None:
    helpers.postprocessing.postprocess_text.side_effect = ValueError("Unsupported spans")
    blocks = adapter._postprocess_parse35_blocks(RAW, table_format="markdown")
    assert blocks[0]["content"] == TABLE
    assert "Keeping HTML" in caplog.text


def _frame() -> pd.DataFrame:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 20)).save(buffer, format="PNG")
    return pd.DataFrame([{"page_image": {"image_b64": base64.b64encode(buffer.getvalue()).decode()}}])


@pytest.mark.parametrize("model_id", [NEMOTRON_PARSE35_MODEL, NEMOTRON_PARSE_V12_MODEL, None])
def test_model_version_selects_prompt_and_postprocessing(helpers: SimpleNamespace, model_id: str | None) -> None:
    model = SimpleNamespace(model_name=model_id, invoke=Mock(return_value=RAW))
    result = nemotron_parse_pages(_frame(), model=model, extract_tables=True, extract_charts=True)
    is35 = model_id == NEMOTRON_PARSE35_MODEL
    assert model.invoke.call_args.kwargs["task_prompt"] == (
        NEMOTRON_PARSE35_DEFAULT_TASK_PROMPT if is35 else NEMOTRON_PARSE_DEFAULT_TASK_PROMPT
    )
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"] is None
    if is35:
        assert result.iloc[0]["table"][0]["text"] == TABLE
        helpers.postprocess_document.assert_called_once_with(RAW)
    else:
        helpers.postprocess_document.assert_not_called()
        assert "<table" not in result.iloc[0]["table"][0]["text"]


@pytest.mark.parametrize("table_format", ["HTML", "simple_html", "markdown", "latex"])
def test_explicit_remote_35_honors_format_and_prompt_override(helpers: SimpleNamespace, table_format: str) -> None:
    client = Mock()
    client.invoke_parse_pages.return_value = [{"content": RAW, "finish_reason": "stop"}]
    result = nemotron_parse_pages(
        _frame(),
        invoke_url="http://parse/v1/parse",
        nim_client=client,
        nemotron_parse_model=NEMOTRON_PARSE35_MODEL,
        nemotron_parse_table_format=table_format,
        task_prompt="custom",
        extract_tables=True,
    )
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"] is None
    assert (
        result.iloc[0]["table"][0]["text"]
        == adapter._postprocess_parse35_blocks(RAW, table_format=table_format)[0]["content"]
    )


def test_actor_retains_explicit_prompt_and_format() -> None:
    actor = NemotronParseGPUActor(task_prompt="custom", nemotron_parse_table_format="simple_html")
    assert actor._task_prompt == "custom"
    assert actor.get_constructor_kwargs()["nemotron_parse_table_format"] == "simple_html"


def test_format_requires_35_and_invalid_choices_fail_before_inference() -> None:
    model = SimpleNamespace(model_name=NEMOTRON_PARSE_V12_MODEL, invoke=Mock())
    with pytest.raises(ValueError, match="Parse 3.5"):
        nemotron_parse_pages(_frame(), model=model, nemotron_parse_table_format="HTML")
    model.invoke.assert_not_called()
    with pytest.raises(ValueError):
        ExtractParams(method="nemotron_parse", nemotron_parse_table_format="csv")
    with pytest.raises(ValueError, match="method='nemotron_parse'"):
        ExtractParams(nemotron_parse_table_format="HTML")
    params = ExtractParams(method="nemotron_parse", nemotron_parse_table_format="simple_html")
    assert params.model_dump()["nemotron_parse_table_format"] == "simple_html"
    for config in (
        {"nemotron_parse_model": NEMOTRON_PARSE_V12_MODEL},
        {"nemotron_parse_invoke_url": "http://parse/v1/parse"},
    ):
        with pytest.raises(ValueError, match="Parse 3.5"):
            ExtractParams(method="nemotron_parse", nemotron_parse_table_format="HTML", **config)


@pytest.mark.parametrize("offline", [False, True])
def test_loader_pins_helpers_without_downloading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    package = tmp_path / "nemotron_parse"
    package.mkdir()
    (package / "__init__.py").write_text("from .submodule import sentinel\n")
    (package / "submodule.py").write_text("sentinel = 35\n")
    revision = get_hf_revision(NEMOTRON_PARSE35_MODEL)
    name = f"_nrl_parse35_postprocessing_{revision}"
    monkeypatch.delitem(sys.modules, name, raising=False)
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", offline)
    monkeypatch.setattr(adapter, "resolve_hf_cache_dir", lambda: "/cache")
    adapter._load_parse35_helpers.cache_clear()
    try:
        module = adapter._load_parse35_helpers()
        assert module.sentinel == 35
        assert adapter._load_parse35_helpers() is module
        download.assert_called_once_with(
            NEMOTRON_PARSE35_MODEL,
            revision=revision,
            cache_dir="/cache",
            allow_patterns=["nemotron_parse/**"],
            local_files_only=offline,
        )
    finally:
        adapter._load_parse35_helpers.cache_clear()
        sys.modules.pop(name, None)
        sys.modules.pop(name + ".submodule", None)


@pytest.mark.parametrize("model_id", [NEMOTRON_PARSE35_MODEL, NEMOTRON_PARSE_V12_MODEL])
def test_local_prompt_override_is_preserved(helpers: SimpleNamespace, model_id: str) -> None:
    model = SimpleNamespace(model_name=model_id, invoke=Mock(return_value=RAW))
    nemotron_parse_pages(_frame(), model=model, task_prompt="custom", extract_tables=True)
    assert model.invoke.call_args.kwargs["task_prompt"] == "custom"


@pytest.mark.parametrize("table_format", ["HTML", "simple_html", "markdown", "latex"])
def test_batch_graph_forwards_table_format(table_format: str) -> None:
    from nemo_retriever.graph.ingestor_runtime import build_graph

    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(method="nemotron_parse", nemotron_parse_table_format=table_format),
    )
    nodes = list(graph.roots)
    while nodes:
        node = nodes.pop()
        if node.operator.__class__.__name__ == "NemotronParseActor":
            assert node.operator_kwargs["nemotron_parse_table_format"] == table_format
            break
        nodes.extend(node.children)
    else:
        pytest.fail("Parse actor missing from graph")


def test_inprocess_graph_forwards_table_format(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_retriever.operators.graph_ops import multi_type_extract_operator as multi
    from nemo_retriever.common.ray_resource_hueristics import Resources

    received = []

    class Identity:
        def __init__(self, **kwargs):
            received.append(kwargs)

        def run(self, frame):
            return frame

    monkeypatch.setattr(multi.DocToPdfConversionActor, "run", lambda self, frame: frame)
    monkeypatch.setattr(multi.PDFSplitActor, "run", lambda self, frame: frame)
    monkeypatch.setattr(multi.PDFExtractionActor, "run", lambda self, frame: frame)
    monkeypatch.setattr(multi, "resolve_operator_class", lambda *args, **kwargs: Identity)
    monkeypatch.setattr(
        "nemo_retriever.common.ray_resource_hueristics.gather_local_resources",
        lambda: Resources(cpu_count=8, gpu_count=1),
    )
    actor = multi.MultiTypeExtractCPUActor(
        extraction_mode="pdf",
        extract_params=ExtractParams(method="nemotron_parse", nemotron_parse_table_format="simple_html"),
    )
    frame = pd.DataFrame({"path": ["example.pdf"]})
    pd.testing.assert_frame_equal(actor._run_pdf_pipeline(frame), frame)
    assert received[0]["nemotron_parse_table_format"] == "simple_html"


@pytest.mark.parametrize("prompt", [None, "custom"])
def test_remote_chat_35_selects_version_prompt(helpers: SimpleNamespace, prompt: str | None) -> None:
    client = Mock()
    client.invoke_chat_completions_images.return_value = [RAW]
    result = nemotron_parse_pages(
        _frame(),
        invoke_url="http://parse/v1/chat/completions",
        nim_client=client,
        nemotron_parse_model=NEMOTRON_PARSE35_MODEL,
        task_prompt=prompt,
        extract_tables=True,
    )
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"] is None
    assert client.invoke_chat_completions_images.call_args.kwargs["task_prompt"] == (
        NEMOTRON_PARSE35_DEFAULT_TASK_PROMPT if prompt is None else prompt
    )
    assert client.invoke_chat_completions_images.call_args.kwargs["model"] == NEMOTRON_PARSE35_MODEL

    assert client.invoke_chat_completions_images.call_args.kwargs["extra_body"] == {
        "max_tokens": 8744,
        "top_k": 1,
        "skip_special_tokens": False,
    }


@pytest.fixture
def interleaved(helpers: SimpleNamespace) -> SimpleNamespace:
    helpers.postprocess_document.side_effect = lambda _: {
        "blocks": [
            {"class": "Text", "bbox": [0, 0, 1, 0.1], "text": "First", "content": "First"},
            {"class": "Table", "bbox": [0, 0.1, 1, 0.4], "text": RAW_TABLE, "content": TABLE},
            {"class": "Text", "bbox": [0, 0.4, 1, 0.5], "text": "Last", "content": "Last"},
            {"class": "Chart", "bbox": [0, 0.5, 1, 0.8], "text": "chart", "content": CHART},
            {"class": "Picture", "bbox": [0, 0.8, 1, 1], "text": "Label", "content": "Label"},
        ]
    }
    model = SimpleNamespace(model_name=NEMOTRON_PARSE35_MODEL, invoke=Mock(return_value=RAW))
    return SimpleNamespace(model=model, helpers=helpers)


def _ordered_result(interleaved: SimpleNamespace, table_format: str = "HTML") -> pd.DataFrame:
    return nemotron_parse_pages(
        _frame(),
        model=interleaved.model,
        nemotron_parse_table_format=table_format,
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )


@pytest.mark.parametrize("table_format", ["HTML", "simple_html"])
def test_page_embedding_preserves_interleaved_blocks(interleaved: SimpleNamespace, table_format: str) -> None:
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = _ordered_result(interleaved, table_format)
    table = result.iloc[0]["table"][0]["text"]
    text = collapse_content_to_page_rows(result).iloc[0]["text"]
    assert text == "\n\n".join(["First", table, "Last", CHART, "Label"])
    assert result.iloc[0]["text"] == "First\n\nLast"


@pytest.mark.parametrize("serialization", ["arrow", "parquet"])
@pytest.mark.parametrize("table_format", ["HTML", "simple_html"])
def test_serialized_parse_preserves_model_order(
    interleaved: SimpleNamespace, table_format: str, serialization: str
) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = _ordered_result(interleaved, table_format)
    table = result.iloc[0]["table"][0]["text"]
    expected = "\n\n".join(["First", table, "Last", CHART, "Label"])
    if serialization == "arrow":
        restored = pa.Table.from_pandas(result).to_pandas()
    else:
        buffer = io.BytesIO()
        result.to_parquet(buffer, index=False)
        buffer.seek(0)
        restored = pd.read_parquet(buffer)

    assert collapse_content_to_page_rows(restored).iloc[0]["text"] == expected
    assert restored.iloc[0]["ordered_page_text"] == expected
    assert expected in to_markdown(restored)
    assert collapse_content_to_page_rows(restored, content_columns=()).iloc[0]["text"] == "First\n\nLast"


@pytest.mark.parametrize(
    "columns,expected",
    [((), ["First", "Last"]), (("table",), ["First", TABLE, "Last"]), (("infographic",), ["First", "Last", "Label"])],
)
def test_ordered_blocks_keep_content_column_selection(
    interleaved: SimpleNamespace, columns: tuple, expected: list
) -> None:
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = _ordered_result(interleaved)
    assert collapse_content_to_page_rows(result, content_columns=columns).iloc[0]["text"] == "\n\n".join(expected)


def test_ordered_blocks_keep_later_captions_and_added_regions(interleaved: SimpleNamespace) -> None:
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = _ordered_result(interleaved)
    result.iloc[0]["chart"][0]["caption"] = "Chart caption"
    result.iloc[0]["infographic"].append({"text": "Added figure"})
    assert collapse_content_to_page_rows(result).iloc[0]["text"] == "\n\n".join(
        ["First", TABLE, "Last", CHART, "Chart caption", "Label", "Added figure"]
    )


def test_ordered_blocks_do_not_replace_custom_text_column(interleaved: SimpleNamespace) -> None:
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = _ordered_result(interleaved)
    result["custom"] = ["Custom text"]
    assert (
        collapse_content_to_page_rows(result, text_column="custom", content_columns=()).iloc[0]["custom"]
        == "Custom text"
    )


def test_real_split_does_not_restore_whole_page(interleaved: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows
    from nemo_retriever.common.modality.txt import split

    monkeypatch.setattr(split, "_get_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(split, "split_text_by_tokens", lambda *args, **kwargs: ["First", "Last"])
    result = split.split_df(_ordered_result(interleaved), max_tokens=1, overlap_tokens=0)
    collapsed = collapse_content_to_page_rows(result)
    assert collapsed["text"].tolist() == ["\n\n".join(["First", TABLE, CHART, "Label"]), "Last"]


@pytest.mark.parametrize("serialization", ["memory", "arrow", "parquet"])
@pytest.mark.parametrize("edit", ["text", "remove_table", "replace_table"])
@pytest.mark.parametrize("collapse_first", [False, True])
def test_ordered_blocks_use_live_edits(
    interleaved: SimpleNamespace, serialization: str, edit: str, collapse_first: bool
) -> None:
    import pyarrow as pa

    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows
    from nemo_retriever.common.io.markdown import to_markdown, to_markdown_by_page

    result = _ordered_result(interleaved)
    if collapse_first:
        result = collapse_content_to_page_rows(result)
    if edit == "text":
        result.at[0, "text"] = "Corrected text"
        expected = ["Corrected text"] if collapse_first else ["Corrected text", TABLE, CHART, "Label"]
        text_only = "Corrected text"
    elif edit == "remove_table":
        result.at[0, "table"] = []
        expected = ["First", "Last", CHART, "Label"]
        text_only = "First\n\nLast"
    else:
        box = result.iloc[0]["table"][0]["bbox_xyxy_norm"]
        result.at[0, "table"] = [{"bbox_xyxy_norm": box, "text": "Corrected table"}]
        expected = ["First", "Corrected table", "Last", CHART, "Label"]
        text_only = "First\n\nLast"
    if serialization == "arrow":
        result = pa.Table.from_pandas(result).to_pandas()
    elif serialization == "parquet":
        buffer = io.BytesIO()
        result.to_parquet(buffer, index=False)
        buffer.seek(0)
        result = pd.read_parquet(buffer)

    expected_text = "\n\n".join(expected)
    assert collapse_content_to_page_rows(result).iloc[0]["text"] == expected_text
    assert to_markdown_by_page(result) == {-1: "## Page Unknown\n\n" + expected_text}
    assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\n" + expected_text
    assert collapse_content_to_page_rows(result, content_columns=()).iloc[0]["text"] == text_only


@pytest.mark.parametrize("serialization", ["memory", "arrow", "parquet"])
@pytest.mark.parametrize("table_format", ["HTML", "simple_html"])
@pytest.mark.parametrize("columns", [("table", "chart", "infographic"), (), ("table",), ("infographic",)])
def test_collapsed_page_markdown_does_not_repeat_content(
    interleaved: SimpleNamespace, serialization: str, table_format: str, columns: tuple
) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown, to_markdown_by_page
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    extracted = _ordered_result(interleaved, table_format)
    blocks = ["First"]
    if "table" in columns:
        blocks.append(extracted.iloc[0]["table"][0]["text"])
    blocks.append("Last")
    if "chart" in columns:
        blocks.append(CHART)
    if "infographic" in columns:
        blocks.append("Label")
    expected = "\n\n".join(blocks)
    result = collapse_content_to_page_rows(extracted, content_columns=columns)
    if serialization == "arrow":
        result = pa.Table.from_pandas(result).to_pandas()
    elif serialization == "parquet":
        buffer = io.BytesIO()
        result.to_parquet(buffer, index=False)
        buffer.seek(0)
        result = pd.read_parquet(buffer)

    assert result.iloc[0]["text"] == expected
    assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\n" + expected
    assert to_markdown_by_page(result) == {-1: "## Page Unknown\n\n" + expected}
    assert collapse_content_to_page_rows(result, content_columns=columns).iloc[0]["text"] == expected
    assert collapse_content_to_page_rows(result, content_columns=()).iloc[0]["text"] == "First\n\nLast"


@pytest.mark.parametrize("serialization", ["memory", "arrow", "parquet"])
@pytest.mark.parametrize("table_format", ["HTML", "simple_html"])
@pytest.mark.parametrize(
    "edit",
    [
        "partial",
        "remove_table_text",
        "replace_table_text",
        "duplicate_table_text",
        "clear",
        "clear_none",
        "clear_nan",
        "clear_na",
    ],
)
def test_edits_to_assembled_page_remain_assembled(
    interleaved: SimpleNamespace, serialization: str, table_format: str, edit: str
) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown, to_markdown_by_page
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = collapse_content_to_page_rows(_ordered_result(interleaved, table_format))
    original = result.iloc[0]["text"]
    table = result.iloc[0]["table"][0]["text"]
    if edit == "partial":
        expected = original.replace("First", "Corrected first")
    elif edit == "remove_table_text":
        expected = original.replace(table + "\n\n", "")
    elif edit == "replace_table_text":
        expected = original.replace(table, "Corrected table")
    elif edit == "duplicate_table_text":
        expected = original + "\n\n" + table
    else:
        expected = ""
    result.at[0, "text"] = expected
    if edit == "clear_none":
        result.at[0, "text"] = None
    elif edit == "clear_nan":
        result.at[0, "text"] = float("nan")
    elif edit == "clear_na":
        result.at[0, "text"] = pd.NA

    for _ in range(3):
        if serialization == "arrow":
            result = pa.Table.from_pandas(result).to_pandas()
        elif serialization == "parquet":
            result = pd.read_parquet(io.BytesIO(result.to_parquet(index=False)))
        page = "## Page Unknown" + ("\n\n" + expected if expected else "\n")
        assert to_markdown(result) == "# Extracted Content\n\n" + page
        assert to_markdown_by_page(result) == {-1: page}
        result = collapse_content_to_page_rows(result)
        assert result.iloc[0]["text"] == expected


@pytest.mark.parametrize("serialization", ["arrow", "parquet"])
def test_mixed_page_text_states_survive_serialization(interleaved: SimpleNamespace, serialization: str) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown_by_page
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    raw = _ordered_result(interleaved)
    collapsed = collapse_content_to_page_rows(raw)
    edited = collapsed.copy(deep=True)
    edited.at[0, "text"] = edited.iloc[0]["text"].replace("First", "Corrected first")
    edited = collapse_content_to_page_rows(edited)
    collapsed.at[0, "_collapsed_page_text"].pop("assembled_text_edited")
    expected = [collapsed.iloc[0]["text"], collapsed.iloc[0]["text"], edited.iloc[0]["text"]]
    result = pd.concat([raw, collapsed, edited], ignore_index=True)
    result["page_number"] = [1, 2, 3]
    if serialization == "arrow":
        result = pa.Table.from_pandas(result).to_pandas()
    else:
        result = pd.read_parquet(io.BytesIO(result.to_parquet(index=False)))

    assert collapse_content_to_page_rows(result)["text"].tolist() == expected
    assert to_markdown_by_page(result) == {
        index: f"## Page {index}\n\n" + text for index, text in enumerate(expected, 1)
    }


def test_split_edited_assembled_page_does_not_append_regions(
    interleaved: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_retriever.common.io.markdown import to_markdown_by_page
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows
    from nemo_retriever.common.modality.txt import split

    result = collapse_content_to_page_rows(_ordered_result(interleaved))
    result.at[0, "text"] = result.iloc[0]["text"].replace("First", "Corrected first")
    chunks = ["Corrected first\n\n" + TABLE, "Last\n\n" + CHART + "\n\nLabel"]
    monkeypatch.setattr(split, "_get_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(split, "split_text_by_tokens", lambda *args, **kwargs: chunks)
    result = split.split_df(result, max_tokens=1, overlap_tokens=0)
    assert collapse_content_to_page_rows(result)["text"].tolist() == chunks
    for index, chunk in enumerate(chunks):
        assert to_markdown_by_page(result.iloc[[index]]) == {-1: "## Page Unknown\n\n" + chunk}


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA])
def test_null_source_text_retains_live_modalities(interleaved: SimpleNamespace, value: object) -> None:
    from nemo_retriever.common.io.markdown import to_markdown
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = _ordered_result(interleaved)
    result.at[0, "text"] = value
    expected = "\n\n".join([TABLE, CHART, "Label"])
    assert collapse_content_to_page_rows(result).iloc[0]["text"] == expected
    assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\n" + expected


def test_projected_collapsed_page_without_text_keeps_live_modalities(interleaved: SimpleNamespace) -> None:
    from nemo_retriever.common.io.markdown import to_markdown
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = collapse_content_to_page_rows(_ordered_result(interleaved)).drop(columns=["text"])
    expected = "\n\n".join([TABLE, CHART, "Label"])
    assert collapse_content_to_page_rows(result).iloc[0]["text"] == expected
    assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\n" + expected


@pytest.mark.parametrize("serialization", ["memory", "arrow", "parquet"])
@pytest.mark.parametrize("table_format", ["HTML", "simple_html"])
@pytest.mark.parametrize("edit", ["remove", "replace", "filter_table", "filter_all", "append"])
@pytest.mark.parametrize("legacy_cache", [False, True])
def test_page_text_edits_reconcile_later_region_changes(
    interleaved: SimpleNamespace, serialization: str, table_format: str, edit: str, legacy_cache: bool
) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = collapse_content_to_page_rows(_ordered_result(interleaved, table_format))
    if legacy_cache:
        result.at[0, "_collapsed_page_text"].pop("content_spans")
        result.at[0, "_collapsed_page_text"].pop("assembled_text_edited")
    table = result.iloc[0]["table"][0]["text"]
    result.at[0, "text"] = result.iloc[0]["text"].replace("First", "Corrected first")
    columns = ("table", "chart", "infographic")
    expected = ["Corrected first", table, "Last", CHART, "Label"]
    if edit == "remove":
        result.at[0, "table"] = []
        expected.remove(table)
    elif edit == "replace":
        result.iloc[0]["table"][0]["text"] = "Corrected table"
        expected[1] = "Corrected table"
    elif edit == "filter_table":
        columns = ("chart", "infographic")
        expected.remove(table)
    elif edit == "filter_all":
        columns = ()
        expected = ["Corrected first", "Last"]
    else:
        result.iloc[0]["table"].append({"text": "Added table"})
        expected.append("Added table")
    expected_text = "\n\n".join(expected)
    for _ in range(3):
        if serialization == "arrow":
            result = pa.Table.from_pandas(result).to_pandas()
        elif serialization == "parquet":
            result = pd.read_parquet(io.BytesIO(result.to_parquet(index=False)))
        result = collapse_content_to_page_rows(result, content_columns=columns)
        assert result.iloc[0]["text"] == expected_text
        assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\n" + expected_text


def test_filtered_edited_page_keeps_equal_native_text(interleaved: SimpleNamespace) -> None:
    from nemo_retriever.common.io.markdown import to_markdown
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = _ordered_result(interleaved)
    result.at[0, "text"] = "First\n\n" + TABLE + "\n\nLast"
    result = collapse_content_to_page_rows(result)
    result.at[0, "text"] = result.iloc[0]["text"].replace("First", "Corrected first")
    result = collapse_content_to_page_rows(result, content_columns=())
    expected = "\n\n".join(["Corrected first", TABLE, "Last"])
    assert result.iloc[0]["text"] == expected
    assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\n" + expected


@pytest.mark.parametrize("serialization", ["memory", "arrow", "parquet"])
@pytest.mark.parametrize("recoverable", [False, True])
def test_legacy_cache_without_ordered_regions_requires_recoverable_content(
    serialization: str, recoverable: bool
) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = collapse_content_to_page_rows(pd.DataFrame([{"text": "Before", "table": [{"text": "TABLE"}]}]))
    result.at[0, "_collapsed_page_text"].pop("content_spans")
    result.at[0, "text"] = result.iloc[0]["text"].replace("Before", "Corrected before")
    if not recoverable:
        result.at[0, "table"] = []
    if serialization == "arrow":
        result = pa.Table.from_pandas(result).to_pandas()
    elif serialization == "parquet":
        result = pd.read_parquet(io.BytesIO(result.to_parquet(index=False)))

    if recoverable:
        result = collapse_content_to_page_rows(result, content_columns=())
        assert result.iloc[0]["text"] == "Corrected before"
        assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\nCorrected before"
    else:
        for render in (collapse_content_to_page_rows, to_markdown):
            with pytest.raises(ValueError, match="Use the original extraction results"):
                render(result)


@pytest.mark.parametrize("serialization", ["memory", "arrow", "parquet"])
@pytest.mark.parametrize("edited", ["Changed TABLE After", "TABLE Before After", "After TABLE Before"])
@pytest.mark.parametrize("legacy_cache", [False, True])
def test_cross_region_rewrites_do_not_silently_bypass_filtering(
    serialization: str, edited: str, legacy_cache: bool
) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown, to_markdown_by_page
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = collapse_content_to_page_rows(pd.DataFrame([{"text": "Before\n\nAfter", "table": [{"text": "TABLE"}]}]))
    if legacy_cache:
        result.at[0, "_collapsed_page_text"].pop("content_spans")
    result.at[0, "text"] = edited
    if serialization == "arrow":
        result = pa.Table.from_pandas(result).to_pandas()
    elif serialization == "parquet":
        result = pd.read_parquet(io.BytesIO(result.to_parquet(index=False)))

    for columns in ((), ("table", "chart", "infographic")):
        with pytest.raises(ValueError, match="cross-region page-text edit"):
            collapse_content_to_page_rows(result, content_columns=columns)
    for render in (to_markdown, to_markdown_by_page):
        with pytest.raises(ValueError, match="cross-region page-text edit"):
            render(result)
    assert result.iloc[0]["text"] == edited


@pytest.mark.parametrize("serialization", ["memory", "arrow", "parquet"])
def test_removed_region_does_not_reclassify_identical_native_text(serialization: str) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    result = collapse_content_to_page_rows(pd.DataFrame([{"text": "TABLE\n\nBefore", "table": [{"text": "TABLE"}]}]))
    result.at[0, "text"] = "TABLE\n\nCorrected before"
    if serialization == "arrow":
        result = pa.Table.from_pandas(result).to_pandas()
    elif serialization == "parquet":
        result = pd.read_parquet(io.BytesIO(result.to_parquet(index=False)))
    result = collapse_content_to_page_rows(result, content_columns=())
    assert result.iloc[0]["text"] == "TABLE\n\nCorrected before"
    assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\nTABLE\n\nCorrected before"


@pytest.mark.parametrize(
    "original,edited",
    [
        ("", ""),
        ("", "new"),
        ("old", ""),
        ("same", "same"),
        ("abc", "xyz"),
        ("abc", "prefixabc"),
        ("abc", "abcsuffix"),
        ("abc", "ac"),
        ("ac", "abc"),
        ("aaaa", "aaaaa"),
        ("aaaaa", "aaaa"),
        ("head\n\nTABLE\n\ntail", "Changed head\n\nTABLE\n\ntail"),
        ("head\n\nTABLE\n\ntail", "head\n\nTABLE\n\nChanged tail"),
        ("head\n\nTABLE\n\ntail", "Changed head\n\nTABLE\n\nChanged tail"),
        ("caf\u00e9\U0001f9eaTABLE", "caf\u00e9\U0001f9eaChanged TABLE"),
    ],
)
def test_trimmed_edits_use_full_page_coordinates(original: str, edited: str) -> None:
    from nemo_retriever.common.modality.content_transforms import _page_edit_opcodes

    left_cursor = right_cursor = 0
    for tag, left, right, edited_left, edited_right in _page_edit_opcodes(original, edited):
        assert (left, edited_left) == (left_cursor, right_cursor)
        assert left <= right <= len(original)
        assert edited_left <= edited_right <= len(edited)
        if tag == "equal":
            assert original[left:right] == edited[edited_left:edited_right]
        elif tag == "insert":
            assert left == right
        elif tag == "delete":
            assert edited_left == edited_right
        left_cursor, right_cursor = right, edited_right
    assert (left_cursor, right_cursor) == (len(original), len(edited))


@pytest.mark.parametrize("position", ["head", "middle", "tail"])
def test_local_edits_skip_long_unchanged_content(monkeypatch: pytest.MonkeyPatch, position: str) -> None:
    import difflib

    from nemo_retriever.common.modality.content_transforms import _page_edit_opcodes

    original = "Before" + "x" * 4000 + "OLD" + "y" * 4000 + "After"
    edited = {
        "head": original.replace("Before", "Corrected before"),
        "middle": original.replace("OLD", "NEW"),
        "tail": original.replace("After", "Corrected after"),
    }[position]
    matcher = difflib.SequenceMatcher
    compared = []

    def capture(junk, left, right, **kwargs):
        compared.append((left, right))
        return matcher(junk, left, right, **kwargs)

    monkeypatch.setattr(difflib, "SequenceMatcher", capture)
    _page_edit_opcodes(original, edited)
    assert len(compared) == 1
    assert max(len(value) for pair in compared for value in pair) < 32


@pytest.mark.parametrize("serialization", ["memory", "arrow", "parquet"])
@pytest.mark.parametrize("edit", ["head", "tail", "table", "before_table", "after_table", "both_ends"])
def test_local_edit_boundaries_preserve_region_filtering(serialization: str, edit: str) -> None:
    import pyarrow as pa

    from nemo_retriever.common.io.markdown import to_markdown
    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows

    record = {
        "text": "Before\n\nAfter",
        "table": [{"text": "TABLE", "bbox_xyxy_norm": [0, 0.2, 1, 0.3]}],
        "ordered_page_text": "Before\n\nTABLE\n\nAfter",
        "ordered_page_text_separator": "\n\n",
        "ordered_regions": [
            {"label_name": "text", "text": "Before"},
            {"label_name": "table", "text": "TABLE", "bbox_xyxy_norm": [0, 0.2, 1, 0.3]},
            {"label_name": "text", "text": "After"},
        ],
    }
    result = collapse_content_to_page_rows(pd.DataFrame([record]))
    original = result.iloc[0]["text"]
    variants = {
        "head": (original.replace("Before", "Corrected before"), "Corrected before\n\nAfter"),
        "tail": (original.replace("After", "Corrected after"), "Before\n\nCorrected after"),
        "table": (original.replace("TABLE", "TA!BLE"), "Before\n\nAfter"),
        "before_table": (original.replace("TABLE", "Note\n\nTABLE"), "Before\n\nNote\n\nAfter"),
        "after_table": (original.replace("TABLE", "TABLE\n\nNote"), "Before\n\nNote\n\nAfter"),
        "both_ends": (
            original.replace("Before", "Corrected before").replace("After", "Corrected after"),
            "Corrected before\n\nCorrected after",
        ),
    }
    result.at[0, "text"], expected = variants[edit]
    if serialization == "arrow":
        result = pa.Table.from_pandas(result).to_pandas()
    elif serialization == "parquet":
        result = pd.read_parquet(io.BytesIO(result.to_parquet(index=False)))
    for _ in range(2):
        result = collapse_content_to_page_rows(result, content_columns=())
        assert result.iloc[0]["text"] == expected
        assert to_markdown(result) == "# Extracted Content\n\n## Page Unknown\n\n" + expected
