# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse configuration reaches inference and postprocessing through public graphs."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nemo_retriever.common.params import ExtractParams, NO_API_KEY, ParseParams
from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL, NEMOTRON_PARSE_V12_MODEL
from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.extract.parse.chart_routing import ChartRoutingCPUActor, ChartRoutingGPUActor
from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseGPUActor, nemotron_parse_pages

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
        parse=config(table_format="markdown", api_key="parse-secret", request_timeout_s=321, **SAMPLING),
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
    assert "api_key" not in params._parse_actor_overrides()
    assert params._parse_actor_overrides()["request_timeout_s"] == params.request_timeout_s
    disabled = ExtractParams(method="nemotron_parse", api_key="explicit-key", parse=ParseParams(api_key=NO_API_KEY))
    assert disabled._parse_actor_overrides()["api_key"] is None
    restored = ExtractParams.model_validate_json(disabled.model_dump_json())
    assert restored._parse_actor_overrides()["api_key"] is None
    assert restored.api_key == "explicit-key"
    monkeypatch.setenv("PARSE_TEST_API_KEY", "parse-key")
    reference = ExtractParams(method="nemotron_parse", parse=ParseParams(api_key="os.environ/PARSE_TEST_API_KEY"))
    assert reference._parse_actor_overrides()["api_key"] == "parse-key"


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
def test_mixed_input_parse_path_renders_and_forwards_group(monkeypatch, mode):
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
            parse=config(table_format="HTML", **SAMPLING),
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


@pytest.mark.parametrize("routing", [False, True])
def test_local_dflash_options_reach_graph_and_model(routing):
    params = ExtractParams(
        method="ocr" if routing else "nemotron_parse",
        chart_routing=routing,
        parse=ParseParams(use_dflash=True, num_speculative_tokens=16),
    )
    graph = build_graph(extraction_mode="pdf", extract_params=params)
    kwargs = get_node_kwargs(graph, "ChartRoutingActor" if routing else "NemotronParseActor")
    assert kwargs["use_dflash"] is True and kwargs["num_speculative_tokens"] == 16
    with patch("nemo_retriever.models.local.NemotronParse35") as factory:
        actor = NemotronParseGPUActor(use_dflash=True, num_speculative_tokens=16)
        actor._ensure_model()
    assert factory.call_args.kwargs["use_dflash"] is True
    assert factory.call_args.kwargs["num_speculative_tokens"] == 16
    assert actor.get_constructor_kwargs()["num_speculative_tokens"] == 16


@pytest.mark.parametrize(
    "config",
    [
        dict(use_dflash=True, invoke_url=URL),
        dict(use_dflash=False, invoke_url=URL),
        dict(num_speculative_tokens=16, invoke_url=URL),
        dict(use_dflash=False, num_speculative_tokens=16),
        dict(use_dflash=True, model=NEMOTRON_PARSE_V12_MODEL),
        dict(num_speculative_tokens=7),
    ],
)
def test_unsupported_dflash_configurations_fail_before_inference(config):
    with pytest.raises(ValueError):
        ExtractParams(method="nemotron_parse", parse=ParseParams(**config))


def test_parse_configuration_schema_describes_all_fields():
    schema = ParseParams.model_json_schema()
    assert all(field.get("description") for field in schema["properties"].values())
    draft = schema["properties"]["num_speculative_tokens"]["description"]
    assert "8, 16, or 32" in draft and "use_dflash=True" in draft
    assert "chart crops" in schema["properties"]["max_tokens"]["description"]
    restored = ParseParams.model_validate({"table_format": "simple_html", "top_k": 50})
    assert restored.table_format == "simple_html" and restored.top_k == 50
