# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse configuration reaches inference and postprocessing through public graphs."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nemo_retriever.common.params import ExtractParams, NO_API_KEY, ParseParams
from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL, build_parse_actor_kwargs
from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.extract.parse.chart_routing import ChartRoutingCPUActor, ChartRoutingGPUActor
from nemo_retriever.operators.extract.parse.nemotron_parse import (
    NEMOTRON_PARSE_REMOTE_DEFAULT_MODEL as NEMOTRON_PARSE_V12_MODEL,
    NemotronParseGPUActor,
    nemotron_parse_pages,
)

URL = "http://localhost:8000/v1/chat/completions"
SAMPLING = dict(temperature=0.2, repetition_penalty=1.05, top_k=50, top_p=1.0, seed=0, max_tokens=8744)


def config(**overrides):
    return ParseParams(model=NEMOTRON_PARSE35_MODEL, invoke_url=URL, **overrides)


@pytest.mark.parametrize("routing", [False, True])
def test_nested_configuration_round_trips_and_reaches_pdf_actor(routing):
    params = ExtractParams(
        method="ocr" if routing else "nemotron_parse",
        chart_routing=routing,
        api_key="ocr-secret",
        parse=config(
            table_format="markdown",
            task_prompt="picture text",
            api_key="parse-secret",
            request_timeout_s=321,
            **SAMPLING,
        ),
    )
    assert "parse-secret" not in repr(params) and "ocr-secret" not in repr(params)
    restored = ExtractParams.model_validate_json(params.model_dump_json())
    assert restored.parse.model_dump() == params.parse.model_dump()
    graph = build_graph(extraction_mode="pdf", extract_params=restored)
    kwargs = get_node_kwargs(graph, "ChartRoutingActor" if routing else "NemotronParseActor")
    assert kwargs["nemotron_parse_invoke_url"] == URL
    assert kwargs["nemotron_parse_table_format"] == "markdown"
    assert kwargs["api_key"] == "parse-secret"
    assert kwargs["request_timeout_s"] == 321
    assert kwargs["parse_sampling_params"] == SAMPLING
    assert kwargs["task_prompt"] == "picture text"
    assert params.api_key == "ocr-secret"
    if routing:
        assert get_node_kwargs(graph, "OCRActor")["api_key"] == "ocr-secret"


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", -0.1),
        ("temperature", float("nan")),
        ("repetition_penalty", 0),
        ("top_k", 0),
        ("top_k", -2),
        ("top_p", 0),
        ("top_p", 1.1),
        ("seed", -1),
        ("max_tokens", 0),
        ("request_timeout_s", 0),
        ("table_format", "csv"),
    ],
)
def test_invalid_parse_options_fail_early(field, value):
    with pytest.raises(ValueError):
        ParseParams(**{field: value})


@pytest.mark.parametrize(
    "nested,flat,value",
    [
        ("model", "nemotron_parse_model", NEMOTRON_PARSE_V12_MODEL),
        ("invoke_url", "nemotron_parse_invoke_url", "http://localhost:8001/v1/chat/completions"),
        ("table_format", "nemotron_parse_table_format", "latex"),
    ],
)
def test_conflicting_legacy_fields_are_rejected(nested, flat, value):
    group = config(table_format="HTML")
    with pytest.raises(ValueError, match=f"parse.{nested} conflicts"):
        ExtractParams(method="nemotron_parse", parse=group, **{flat: value})


def test_sampling_requires_parse35_and_supported_transport():
    with pytest.raises(ValueError, match="sampling overrides require model"):
        ExtractParams(method="nemotron_parse", parse=ParseParams(model=NEMOTRON_PARSE_V12_MODEL, temperature=0))
    with pytest.raises(ValueError, match="sampling overrides require model"):
        ExtractParams(method="nemotron_parse", parse=ParseParams(invoke_url=URL, temperature=0))
    for endpoint in ("http://localhost:8000/v1/parse", "http://localhost:8000/v1/infer"):
        with pytest.raises(ValueError, match="sampling overrides require local inference"):
            ExtractParams(
                method="nemotron_parse",
                parse=ParseParams(
                    model=NEMOTRON_PARSE35_MODEL,
                    invoke_url=endpoint,
                    temperature=0,
                ),
            )
    with pytest.raises(ValueError, match="requires method"):
        ExtractParams(parse=ParseParams(temperature=0))


def test_parse_auth_inherits_or_disables_only_parse_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "environment-secret")
    group = ParseParams()
    assert group.api_key is None
    params = ExtractParams(method="nemotron_parse", api_key="explicit-key", parse=group)
    assert build_parse_actor_kwargs(params)["api_key"] == "explicit-key"
    assert build_parse_actor_kwargs(params)["request_timeout_s"] == params.request_timeout_s
    disabled = ExtractParams(method="nemotron_parse", api_key="explicit-key", parse=ParseParams(api_key=NO_API_KEY))
    assert build_parse_actor_kwargs(disabled)["api_key"] is None
    restored = ExtractParams.model_validate_json(disabled.model_dump_json())
    assert build_parse_actor_kwargs(restored)["api_key"] is None
    assert restored.api_key == "explicit-key"
    monkeypatch.setenv("PARSE_TEST_API_KEY", "parse-key")
    reference = ExtractParams(method="nemotron_parse", parse=ParseParams(api_key="os.environ/PARSE_TEST_API_KEY"))
    assert build_parse_actor_kwargs(reference)["api_key"] == "parse-key"


def test_local_fullpage_adapter_receives_explicit_sampling():
    with patch("nemo_retriever.models.local.NemotronParse35") as factory:
        actor = NemotronParseGPUActor(parse_sampling_params=SAMPLING)
        actor._ensure_model()
    kwargs = factory.call_args.kwargs
    assert kwargs["max_tokens"] == 8744
    assert kwargs["sampling_params"] == {k: v for k, v in SAMPLING.items() if k != "max_tokens"}
    assert actor.get_constructor_kwargs()["parse_sampling_params"] == SAMPLING


def test_remote_fullpage_sends_zero_temperature_and_all_overrides(monkeypatch):
    from nemo_retriever.operators.extract.parse import nemotron_parse as module

    monkeypatch.setattr(module, "_route_parsed_elements", lambda *a, **kw: ([], [], [], "text"))
    client = Mock()
    client.invoke_chat_completions_images.return_value = ["raw"]
    sampling = {**SAMPLING, "temperature": 0, "top_k": -1, "seed": 0, "max_tokens": 12}
    result = nemotron_parse_pages(
        pd.DataFrame([{"page_image": {"image_b64": "encoded"}}]),
        invoke_url=URL,
        nemotron_parse_model=NEMOTRON_PARSE35_MODEL,
        nim_client=client,
        parse_sampling_params=sampling,
    )
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"] is None
    kwargs = client.invoke_chat_completions_images.call_args.kwargs
    assert kwargs["temperature"] == 0 and kwargs["repetition_penalty"] == 1.05
    assert kwargs["extra_body"] == dict(top_k=-1, top_p=1, seed=0, max_tokens=12, skip_special_tokens=False)


@pytest.mark.parametrize("table_format", ["HTML", "simple_html", "markdown", "latex"])
def test_crop_actor_passes_format_and_sampling_and_can_be_reconstructed(table_format):
    with patch("nemo_retriever.operators.extract.parse.chart_routing.route_chart_regions") as route:
        actor = ChartRoutingCPUActor(
            nemotron_parse_invoke_url=URL,
            nemotron_parse_table_format=table_format,
            parse_sampling_params={"temperature": 0, "max_tokens": 99},
        )
        clone = ChartRoutingCPUActor(**actor.get_constructor_kwargs())
        clone.process(pd.DataFrame())
    kwargs = route.call_args.kwargs
    assert kwargs["table_format"] == table_format
    assert kwargs["sampling_params"] == dict(
        temperature=0,
        repetition_penalty=1,
        top_k=50,
        top_p=1,
        seed=0,
        max_tokens=99,
    )


def test_local_crop_actor_uses_crop_defaults_and_overrides():
    with (
        patch("nemo_retriever.models.local.nemotron_parse35.NemotronParse35") as factory,
        patch("nemo_retriever.operators.extract.parse.chart_routing.route_chart_regions"),
    ):
        actor = ChartRoutingGPUActor(parse_sampling_params={"top_k": -1, "max_tokens": 123})
        actor.process(pd.DataFrame())
    assert factory.call_args.kwargs["max_tokens"] == 123
    assert factory.call_args.kwargs["sampling_params"] == dict(
        temperature=0.1,
        repetition_penalty=1,
        top_k=-1,
        top_p=1,
        seed=0,
    )


@pytest.mark.parametrize("mode", ["pdf", "image"])
@pytest.mark.parametrize("parse_mode", ["single", "double", "triple"])
def test_mixed_input_parse_path_renders_and_forwards_group(monkeypatch, mode, parse_mode):
    from nemo_retriever.operators.graph_ops import multi_type_extract_operator as module

    events, received = [], []
    for cls, name in (
        (module.DocToPdfConversionActor, "convert"),
        (module.PDFSplitActor, "split"),
        (module.PDFExtractionActor, "render"),
        (module.ImageLoadActor, "load"),
    ):

        def run(self, data, stage=name):
            events.append(stage)
            return data

        monkeypatch.setattr(cls, "run", run)
    actor = module.MultiTypeExtractCPUActor(
        extraction_mode=mode,
        extract_params=ExtractParams(
            method="nemotron_parse",
            parse=config(table_format="HTML", mode=parse_mode, **SAMPLING),
        ),
    )

    def instantiate(cls, **kwargs):
        events.append(cls.__name__)
        received.append(kwargs)
        return Mock(run=lambda data: data)

    monkeypatch.setattr(actor, "_instantiate_resolved", instantiate)
    frame = pd.DataFrame({"path": ["example.pdf" if mode == "pdf" else "example.png"]})
    result = getattr(actor, f"_run_{mode}_pipeline")(frame)
    pd.testing.assert_frame_equal(result, frame)
    assert events == (
        ["convert", "split", "render", "NemotronParseActor"] if mode == "pdf" else ["load", "NemotronParseActor"]
    )
    assert received[0]["parse_sampling_params"] == SAMPLING
    assert received[0]["nemotron_parse_table_format"] == "HTML"
    assert received[0].get("parse_mode", "single") == parse_mode


@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("tokens", ["auto", None, 1, 16])
def test_main_dflash_option_reaches_both_paths(routing, tokens):
    params = ExtractParams(
        method="ocr" if routing else "nemotron_parse",
        chart_routing=routing,
        nemotron_parse_dflash_tokens=tokens,
    )
    graph = build_graph(extraction_mode="pdf", extract_params=params)
    kwargs = get_node_kwargs(graph, "ChartRoutingActor" if routing else "NemotronParseActor")
    assert kwargs["nemotron_parse_dflash_tokens"] == tokens
    actor_class = ChartRoutingGPUActor if routing else NemotronParseGPUActor
    target = (
        "nemo_retriever.models.local.nemotron_parse35.NemotronParse35"
        if routing
        else "nemo_retriever.models.local.NemotronParse35"
    )
    original = actor_class(nemotron_parse_dflash_tokens=tokens)
    actor = actor_class(**original.get_constructor_kwargs())
    with patch(target) as factory:
        if routing:
            actor.process(pd.DataFrame())
        else:
            actor._ensure_model()
    expected = tokens if routing or tokens != "auto" else 32
    assert factory.call_args.kwargs["dflash_tokens"] == expected
    if routing:
        actor.cleanup()


@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("tokens", [0, 33, True, 1.5])
def test_invalid_main_dflash_setting_is_rejected(routing, tokens):
    with pytest.raises(ValueError):
        ExtractParams(
            method="ocr" if routing else "nemotron_parse", chart_routing=routing, nemotron_parse_dflash_tokens=tokens
        )


@pytest.mark.parametrize("tokens", ["auto", None, 16])
def test_remote_routing_uses_main_decoding_validation(tokens):
    if isinstance(tokens, int):
        with pytest.raises(ValueError, match="requires local Nemotron Parse 3.5"):
            ExtractParams(
                method="ocr", chart_routing=True, nemotron_parse_invoke_url=URL, nemotron_parse_dflash_tokens=tokens
            )
        with pytest.raises(ValueError, match="requires local Nemotron Parse 3.5"):
            ChartRoutingCPUActor(nemotron_parse_invoke_url=URL, nemotron_parse_dflash_tokens=tokens)
    else:
        params = ExtractParams(
            method="ocr", chart_routing=True, nemotron_parse_invoke_url=URL, nemotron_parse_dflash_tokens=tokens
        )
        graph = build_graph(extraction_mode="pdf", extract_params=params)
        kwargs = get_node_kwargs(graph, "ChartRoutingActor")
        assert kwargs["nemotron_parse_dflash_tokens"] == tokens


def test_parse_configuration_schema_describes_all_fields():
    schema = ParseParams.model_json_schema()
    assert all(field.get("description") for field in schema["properties"].values())
    assert "use_dflash" not in schema["properties"]
    assert "num_speculative_tokens" not in schema["properties"]
    assert "chart crops" in schema["properties"]["max_tokens"]["description"]
    restored = ParseParams.model_validate({"table_format": "simple_html", "top_k": 50})
    assert restored.table_format == "simple_html" and restored.top_k == 50


@pytest.mark.parametrize("nested", [False, True])
def test_blank_parse_endpoint_keeps_remote_fallback(nested):
    kwargs = {"parse": ParseParams(invoke_url="  \n")} if nested else {"nemotron_parse_invoke_url": "  \n"}
    params = ExtractParams(method="nemotron_parse", invoke_url=URL, **kwargs)
    graph = build_graph(extraction_mode="pdf", extract_params=params)
    options = get_node_kwargs(graph, "NemotronParseActor")
    assert options.get("invoke_url") == URL
    assert not options.get("nemotron_parse_invoke_url")


def test_blank_routing_endpoint_selects_local_gpu():
    params = ExtractParams(method="ocr", chart_routing=True, nemotron_parse_invoke_url=" \n ")
    assert params.nemotron_parse_invoke_url is None
    from nemo_retriever.operators.extract.parse.chart_routing import ChartRoutingActor

    assert not ChartRoutingActor.prefers_cpu_variant({"nemotron_parse_invoke_url": " \n "})
    with pytest.raises(ValueError, match="requires nemotron_parse_invoke_url"):
        ChartRoutingCPUActor(nemotron_parse_invoke_url=" \n ")


def test_explicit_dflash_cannot_be_silently_lost_on_cpu_fallback():
    from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseCPUActor

    with pytest.raises(ValueError, match="requires local Nemotron Parse 3.5"):
        NemotronParseCPUActor(nemotron_parse_dflash_tokens=16)


@pytest.mark.parametrize("routing", [False, True])
def test_parse_picture_text_is_explicit_and_default_prompt_is_unchanged(routing):
    from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_DEFAULT_TASK_PROMPT

    default = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
    enabled = default.replace("predict_no_text_in_pic", "predict_text_in_pic")
    params = ExtractParams(method="ocr" if routing else "nemotron_parse", chart_routing=routing)
    assert NEMOTRON_PARSE35_DEFAULT_TASK_PROMPT == default
    assert "task_prompt" not in build_parse_actor_kwargs(params, chart_routing=routing)
    params = ExtractParams(method=params.method, chart_routing=routing, parse=ParseParams(task_prompt=enabled))
    kwargs = build_parse_actor_kwargs(params, chart_routing=routing)
    assert kwargs["task_prompt"] == enabled
    actor = ChartRoutingGPUActor(**kwargs) if routing else NemotronParseGPUActor(**kwargs)
    assert (actor.task_prompt if routing else actor._task_prompt) == enabled


@pytest.mark.parametrize("mode", ["double", "triple"])
def test_multipass_mode_round_trip_and_graph_wiring(mode):
    params = ExtractParams(method="nemotron_parse", parse=config(mode=mode))
    restored = ExtractParams.model_validate_json(params.model_dump_json())
    assert restored.parse.mode == mode
    graph = build_graph(extraction_mode="pdf", extract_params=restored)
    assert get_node_kwargs(graph, "NemotronParseActor")["parse_mode"] == mode


@pytest.mark.parametrize(
    "group,extract",
    [
        ({"mode": "quadruple"}, {}),
        ({"mode": "double", "model": NEMOTRON_PARSE_V12_MODEL}, {}),
        ({"mode": "double", "invoke_url": URL}, {}),
        ({"mode": "double", "task_prompt": "custom"}, {}),
        ({"mode": "triple", "model": NEMOTRON_PARSE35_MODEL, "invoke_url": "http://local/v1/parse"}, {}),
        (
            {
                "mode": "triple",
                "model": NEMOTRON_PARSE35_MODEL,
                "invoke_url": "https://integrate.api.nvidia.com/v1/chat/completions",
            },
            {},
        ),
        ({"mode": "double"}, {"method": "ocr", "chart_routing": True}),
    ],
)
def test_invalid_multipass_configuration_fails_early(group, extract):
    with pytest.raises(ValueError):
        ExtractParams(**{"method": "nemotron_parse", **extract}, parse=ParseParams(**group))


def test_single_mode_preserves_existing_actor_configuration():
    assert ParseParams().mode == "single"
    implicit = ExtractParams(method="nemotron_parse", parse=ParseParams())
    explicit = ExtractParams(method="nemotron_parse", parse=ParseParams(mode="single"))
    assert build_parse_actor_kwargs(implicit) == build_parse_actor_kwargs(explicit)
    assert "parse_mode" not in build_parse_actor_kwargs(explicit)
