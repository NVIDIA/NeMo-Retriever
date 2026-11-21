# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-pass adapter contracts without model weights or network access."""

import base64
from copy import deepcopy
from dataclasses import dataclass
import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from nemo_retriever.common.modality.parse import nemotron_parse35_postprocessing as postprocessing
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.operators.extract.parse import nemotron_parse as operator

URL = "http://parse-a/v1/chat/completions"
RAW = "<x_0.1><y_0.2>page<x_0.9><y_0.8><class_Text>"


@dataclass
class Settings:
    temperature: float = 0.2
    repetition_penalty: float = 1.05
    max_tokens: int = 8744
    top_k: int = 50
    top_p: float = 1.0
    seed: int = 0


class GenerationError(RuntimeError):
    def __init__(self, stage, response):
        self.stage, self.response = stage, response
        super().__init__(f"{stage} generation failed: {response}")


@pytest.fixture
def hf(monkeypatch):
    """Mock HF orchestration; real geometry/merge parity is covered by the cached integration below."""
    parser = Mock()
    module = SimpleNamespace(
        DocumentParser=Mock(return_value=parser), GenerationSettings=Settings, GenerationError=GenerationError
    )
    monkeypatch.setattr(postprocessing, "_load_parse35_helpers", lambda: SimpleNamespace(__name__="pinned_helpers"))
    importer = Mock(return_value=module)
    monkeypatch.setattr(operator, "import_module", importer)
    module.parser = parser
    module.importer = importer
    return module


def run(images=None, **kwargs):
    options = dict(
        mode="triple",
        model=None,
        model_name=NEMOTRON_PARSE35_MODEL,
        invoke_url="",
        api_key=None,
        timeout_s=32,
        retry=RemoteRetryParams(),
        sampling={},
    )
    options.update(kwargs)
    return operator._parse35_multipass(
        images if images is not None else [np.zeros((20, 16, 3), dtype=np.uint8)], **options
    )


def test_local_delegates_in_input_order_with_fullpage_only_overrides(hf):
    model = Mock()
    response = {"raw_out": RAW, "finish_reason": "stop", "usage": {"completion_tokens": 12}}
    model.invoke_response.return_value = response
    hf.parser.parse.side_effect = [{"raw_out": "first"}, {"raw_out": "second"}]
    images = [np.zeros((20, 16, 3), dtype=np.uint8), np.zeros((30, 25, 3), dtype=np.uint8)]
    assert run(images, model=model, sampling={"temperature": 0, "max_tokens": 123}) == [
        {"raw_out": "first"},
        {"raw_out": "second"},
    ]
    hf.importer.assert_called_once_with("pinned_helpers.inference")
    config = hf.DocumentParser.call_args.kwargs
    assert config["fullpage"] == Settings(temperature=0, max_tokens=123)
    assert set(config) == {"model", "backend", "fullpage"}
    assert [call.args[0].size for call in hf.parser.parse.call_args_list] == [(16, 20), (25, 30)]
    assert all(call.kwargs == {"mode": "triple"} for call in hf.parser.parse.call_args_list)
    hf.parser.parse_many.assert_not_called()
    crop_settings = Settings(temperature=0.1, repetition_penalty=1.0)
    assert (
        config["backend"](Image.new("RGB", (8, 9)), prompt="crop-prompt", stage="crop", settings=crop_settings)
        == response
    )
    assert model.invoke_response.call_args.kwargs == {
        "task_prompt": "crop-prompt",
        "sampling_params": vars(crop_settings),
    }


def test_remote_preserves_response_settings_transport_and_endpoint_rotation(hf, monkeypatch):
    request = Mock(
        return_value={
            "choices": [{"message": {"content": RAW}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 4096},
        }
    )
    monkeypatch.setattr(operator, "invoke_chat_completion_step", request)
    retry = RemoteRetryParams(remote_max_pool_workers=3, remote_max_retries=2, remote_max_429_retries=1)
    run(invoke_url=URL + ",http://parse-b/v1/chat/completions", api_key="test-key", retry=retry)
    hf.parser.parse_many.assert_called_once()
    assert hf.parser.parse_many.call_args.kwargs == {"mode": "triple", "concurrency": 3}
    hf.parser.parse.assert_not_called()
    backend = hf.DocumentParser.call_args.kwargs["backend"]
    settings = Settings(max_tokens=4096)
    for _ in range(3):
        result = backend(Image.new("RGB", (17, 19)), prompt="detect-prompt", stage="detection", settings=settings)
        assert result == {"raw_out": RAW, "finish_reason": "length", "usage": {"completion_tokens": 4096}}
    calls = [call.kwargs for call in request.call_args_list]
    assert [call["invoke_url"] for call in calls] == [URL, "http://parse-b/v1/chat/completions", URL]
    last = calls[-1]
    assert last["api_key"] == "test-key" and last["timeout_s"] == 32
    assert last["max_retries"] == 2 and last["max_429_retries"] == 1
    assert last["temperature"] == 0.2 and last["max_tokens"] == 4096
    assert last["extra_body"] == dict(repetition_penalty=1.05, top_k=50, top_p=1.0, seed=0, skip_special_tokens=False)
    content = last["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "detect-prompt"}
    with Image.open(io.BytesIO(base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]))) as image:
        assert image.size == (17, 19)
    assert settings == Settings(max_tokens=4096)


@pytest.mark.parametrize("failure", [RuntimeError("unavailable"), {"choices": []}])
def test_remote_failures_retain_stage(hf, monkeypatch, failure):
    request = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    monkeypatch.setattr(operator, "invoke_chat_completion_step", request)
    run(invoke_url=URL)
    backend = hf.DocumentParser.call_args.kwargs["backend"]
    with pytest.raises(GenerationError) as error:
        backend(Image.new("RGB", (16, 20)), prompt="p", stage="crop", settings=Settings())
    assert error.value.stage == "crop"
    assert error.value.response["error"]


def frame(count=1):
    stream = io.BytesIO()
    Image.new("RGB", (16, 20)).save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    return pd.DataFrame([{"page_image": {"image_b64": encoded}, "text": "original"}] * count)


@pytest.mark.parametrize("mode", ["double", "triple"])
@pytest.mark.parametrize("model_name", [NEMOTRON_PARSE35_MODEL, "Nemotron-3.5-Parse"])
def test_pipeline_routes_merged_content_and_records_metadata(monkeypatch, mode, model_name):
    document = {
        "raw_out": RAW,
        "inference": {"mode": mode, "requests": 3, "sampling": {}},
        "blocks": [
            {"class": "Text", "bbox": [0, 0, 1, 0.1], "content": "merged heading"},
            {"class": "Table", "bbox": [0.1, 0.2, 0.9, 0.8], "content": "merged table"},
        ],
    }
    monkeypatch.setattr(operator, "_parse35_multipass", Mock(return_value=[document]))
    render = Mock(side_effect=AssertionError("Merged output has already been rendered"))
    monkeypatch.setattr(postprocessing, "_load_parse35_helpers", lambda: SimpleNamespace(postprocess_document=render))
    output = operator.nemotron_parse_pages(
        frame(), model=Mock(model_name=model_name), parse_mode=mode, extract_text=True, extract_tables=True
    )
    render.assert_not_called()
    assert output.iloc[0]["text"] == "merged heading"
    assert output.iloc[0]["table"][0]["text"] == "merged table"
    assert output.iloc[0]["ordered_page_text"] == "merged heading\n\nmerged table"
    assert output.iloc[0]["table_parse"] == output.iloc[0]["table"]
    meta = output.iloc[0]["nemotron_parse_v1_2"]
    assert meta["error"] is None
    assert meta["multipass"]["helper_revision"] == get_hf_revision(NEMOTRON_PARSE35_MODEL)
    assert meta["multipass"]["mode"] == mode


def test_failed_stage_preserves_batch_failure_contract(monkeypatch):
    monkeypatch.setattr(
        operator,
        "_parse35_multipass",
        Mock(side_effect=GenerationError("crop", {"raw_out": "partial", "finish_reason": "length"})),
    )
    output = operator.nemotron_parse_pages(
        frame(2), model=Mock(model_name=NEMOTRON_PARSE35_MODEL), parse_mode="double", extract_text=True
    )
    assert output.text.tolist() == ["original", "original"]
    for meta in output.nemotron_parse_v1_2:
        assert meta["error"]["parse_mode"] == "double"
        assert meta["error"]["inference_stage"] == "crop"
        assert meta["error"]["finish_reason"] == "length"
        assert "multipass" not in meta


@pytest.mark.parametrize("actor", [operator.NemotronParseCPUActor, operator.NemotronParseGPUActor])
def test_custom_prompt_rejected_before_inference(actor):
    with pytest.raises(ValueError, match="custom task_prompt"):
        actor(
            nemotron_parse_model=NEMOTRON_PARSE35_MODEL,
            nemotron_parse_invoke_url=URL,
            parse_mode="triple",
            task_prompt="custom",
        )


@pytest.fixture
def cached_helpers(monkeypatch):
    """Use real pinned HF code only when its snapshot is already cached; never download in tests."""
    from huggingface_hub import constants
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    postprocessing._load_parse35_helpers.cache_clear()
    try:
        helpers = postprocessing._load_parse35_helpers()
    except (LocalEntryNotFoundError, FileNotFoundError):
        pytest.skip("Pinned Parse 3.5 helper snapshot is not cached")
    yield helpers
    postprocessing._load_parse35_helpers.cache_clear()


@pytest.mark.parametrize("mode", ["double", "triple"])
@pytest.mark.parametrize("structural", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_cached_hf_parity(cached_helpers, monkeypatch, mode, structural, remote):
    """Exact HF merge/geometry parity, including parser-only and detector-only regions."""
    from importlib import import_module

    hf = import_module(f"{cached_helpers.__name__}.inference")
    table = r"\begin{tabular}{cc} A & B \\ 1 & 2 \\ \end{tabular}"
    full = RAW
    if structural:
        full += f"<x_0.1><y_0.2>{table}<x_0.4><y_0.5><class_Table>"
    detect = "<x_0.6><y_0.6><x_0.9><y_0.9><class_Table>" if structural else RAW
    crop = f"<x_0.1><y_0.1>{table}<x_0.9><y_0.9><class_Table>"
    calls = []

    def backend(image, *, prompt, stage, settings):
        calls.append((stage, image.size, settings))
        return {"raw_out": {"fullpage": full, "detection": detect, "crop": crop}[stage], "finish_reason": "stop"}

    expected = hf.DocumentParser(backend=backend).parse(Image.new("RGB", (800, 1000)), mode=mode)
    expected_calls = list(calls)
    calls.clear()

    def invoke(image, *, task_prompt, sampling_params):
        stage = "detection" if task_prompt == hf.DETECT_PROMPT else "fullpage" if image.size == (800, 1000) else "crop"
        return backend(image, prompt=task_prompt, stage=stage, settings=hf.GenerationSettings(**sampling_params))

    if remote:

        def request(**kwargs):
            content = kwargs["messages"][0]["content"]
            encoded = content[1]["image_url"]["url"].split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                settings = {k: v for k, v in kwargs["extra_body"].items() if k != "skip_special_tokens"}
                response = invoke(
                    image,
                    task_prompt=content[0]["text"],
                    sampling_params={
                        **settings,
                        "temperature": kwargs["temperature"],
                        "max_tokens": kwargs["max_tokens"],
                    },
                )
            return {"choices": [{"message": {"content": response["raw_out"]}, "finish_reason": "stop"}]}

        monkeypatch.setattr(operator, "invoke_chat_completion_step", request)
    actual = run(
        [np.zeros((1000, 800, 3), dtype=np.uint8)],
        mode=mode,
        invoke_url=URL if remote else "",
        model=SimpleNamespace(invoke_response=invoke),
    )[0]
    assert calls == expected_calls
    assert actual["raw_out"] == expected["raw_out"]
    assert actual["blocks"] == expected["blocks"]
    for table_format in ("HTML", "simple_html", "markdown", "latex"):
        opts = dict(
            extract_tables=True, extract_charts=True, extract_infographics=True, parse35=True, table_format=table_format
        )
        assert operator._route_parsed_elements(actual["raw_out"], parse35_document=deepcopy(actual), **opts) == (
            operator._route_parsed_elements(actual["raw_out"], **opts)
        )
    assert actual["inference"]["requests"] == (1 + (mode == "triple") + (1 + (mode == "triple") if structural else 0))
    assert actual["inference"]["merge_policy"] == "alternatives"


@pytest.mark.parametrize(
    "response",
    [
        {"raw_out": RAW, "finish_reason": "length"},
        {"raw_out": "", "finish_reason": "stop"},
        {"raw_out": None, "finish_reason": "stop"},
        {"raw_out": RAW, "finish_reason": "content_filter"},
    ],
)
def test_cached_hf_rejects_incomplete_generations(cached_helpers, response):
    from importlib import import_module

    hf = import_module(f"{cached_helpers.__name__}.inference")
    with pytest.raises(hf.GenerationError):
        run(model=SimpleNamespace(invoke_response=lambda *a, **kw: response))


@pytest.mark.parametrize("actor_class", [operator.NemotronParseCPUActor, operator.NemotronParseGPUActor])
@pytest.mark.parametrize("mode", ["double", "triple"])
def test_rebuilt_actor_forwards_mode_to_page_processing(monkeypatch, actor_class, mode):
    from nemo_retriever.graph.pipeline_graph import Node

    process = Mock(return_value=frame())
    monkeypatch.setattr(operator, "nemotron_parse_pages", process)
    actor = actor_class(nemotron_parse_model=NEMOTRON_PARSE35_MODEL, nemotron_parse_invoke_url=URL, parse_mode=mode)
    node = Node(actor)
    rebuilt = node.operator_class(**node.operator_kwargs)
    rebuilt.process(frame())
    assert process.call_args.kwargs["parse_mode"] == mode
    assert process.call_args.kwargs["nemotron_parse_model"] == NEMOTRON_PARSE35_MODEL
    assert process.call_args.kwargs["task_prompt"] is None


def test_remote_callback_propagates_caller_context(hf, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import ContextVar

    trace = ContextVar("multipass_test_trace", default=None)
    observed = []

    def request(**kwargs):
        observed.append(trace.get())
        return {"choices": [{"message": {"content": RAW}, "finish_reason": "stop"}]}

    monkeypatch.setattr(operator, "invoke_chat_completion_step", request)
    token = trace.set("caller-trace")
    try:
        run(invoke_url=URL)
        callback = hf.DocumentParser.call_args.kwargs["backend"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(
                callback, Image.new("RGB", (16, 20)), prompt="p", stage="fullpage", settings=Settings()
            ).result()
    finally:
        trace.reset(token)
    assert observed == ["caller-trace"]
