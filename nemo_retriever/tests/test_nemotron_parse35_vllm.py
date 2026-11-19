# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline contracts for plugin MTP and DFlash2 Parse inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
import itertools
import os
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image
import numpy as np
import pytest
import torch

from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.models.inference import nemotron_parse35_vllm as plugin_module
from nemo_retriever.models.local import nemotron_parse35 as adapter_module
from nemo_retriever.models.local.nemotron_parse35 import NemotronParse35, ParsePageResult


@pytest.fixture(autouse=True)
def model_registry(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Keep plugin registration checks independent of the optional vLLM install."""
    module = ModuleType("vllm.model_executor.models")
    module.ModelRegistry = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", module)
    return module.ModelRegistry


@pytest.fixture
def vllm_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Replace vLLM and network I/O, without importing CUDA model classes."""
    engine = MagicMock()
    engine.generate.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(text=" <class_Text>page ", finish_reason="stop", token_ids=[1, 2])])
    ]
    llm = MagicMock(return_value=engine)
    sampling = MagicMock()
    module = ModuleType("vllm")
    monkeypatch.setattr(adapter_module, "_AsyncParseEngine", llm)
    module.SamplingParams = sampling
    monkeypatch.setitem(sys.modules, "vllm", module)
    register = MagicMock()
    monkeypatch.setattr(plugin_module, "register_nemotron_parse35_vllm", register)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", MagicMock())
    monkeypatch.setattr(adapter_module, "configure_global_hf_cache_base", lambda _: "/cache")
    snapshot = tmp_path / get_hf_revision(NEMOTRON_PARSE35_MODEL)
    draft = snapshot / "dflash2"
    draft.mkdir(parents=True, exist_ok=True)
    (draft / "config.json").write_text("{}")
    (draft / "model.safetensors").touch()
    snapshot = str(snapshot)
    download = MagicMock(return_value=snapshot)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setenv(plugin_module.PARSE35_SNAPSHOT_ENV, snapshot)
    return SimpleNamespace(engine=engine, llm=llm, sampling=sampling, register=register, download=download)


@pytest.mark.parametrize("tokens", ["auto", None, 16])
def test_vllm_pins_plugin_and_weights_and_uses_native_prompt(
    vllm_runtime: SimpleNamespace, tokens: str | int | None
) -> None:
    parser = NemotronParse35(max_tokens=123, dflash_tokens=tokens)
    assert parser.invoke_batch([]) == []
    vllm_runtime.download.assert_not_called()
    assert parser.invoke(Image.new("RGB", (16, 20))) == "<class_Text>page"
    vllm_runtime.register.assert_called_once()
    assert vllm_runtime.download.call_args.kwargs["revision"] == get_hf_revision(NEMOTRON_PARSE35_MODEL)
    assert vllm_runtime.llm.call_args.kwargs["model"] == vllm_runtime.download.return_value
    assert vllm_runtime.llm.call_args.kwargs["enable_prefix_caching"] is False
    assert vllm_runtime.sampling.call_args.kwargs["skip_special_tokens"] is False
    assert vllm_runtime.sampling.call_args.kwargs["max_tokens"] == 123
    request = vllm_runtime.engine.generate.call_args.args[0][0]
    assert request["prompt"] == "<pad>" + parser._task_prompt
    assert request["multi_modal_data"]["image"].size == (16, 20)
    assert vllm_runtime.engine.generate.call_args.kwargs["tokenization_kwargs"] == {"add_special_tokens": False}


def test_vllm_reports_length_limited_generation(vllm_runtime: SimpleNamespace) -> None:
    vllm_runtime.engine.generate.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(text=" capped ", finish_reason="length", token_ids=[1, 2, 3])])
    ]
    parser = NemotronParse35()
    assert parser.invoke_batch_results([Image.new("RGB", (16, 20))]) == [
        ParsePageResult(text="capped", finish_reason="length", output_tokens=3)
    ]
    assert parser.invoke_batch_results([]) == []


@pytest.mark.parametrize("tokens", [None, 32])
def test_vllm_converts_then_submits_batch(
    vllm_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tokens: int | None
) -> None:
    parser = NemotronParse35(backend="vllm", dflash_tokens=tokens)
    events = []
    original_preprocess = parser.preprocess

    def preprocess(page: Image.Image) -> Image.Image:
        events.append("convert")
        return original_preprocess(page)

    def generate(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        events.append("generate")
        return [SimpleNamespace(outputs=[SimpleNamespace(text="page", finish_reason="stop", token_ids=[1])])] * len(
            args[0]
        )

    monkeypatch.setattr(parser, "preprocess", preprocess)
    vllm_runtime.engine.generate.side_effect = generate
    assert parser.invoke_batch([Image.new("RGB", (16, 16))] * 2, task_prompt="custom") == ["page", "page"]
    assert events == ["convert", "convert", "generate"]
    assert vllm_runtime.engine.generate.call_args.args[0][0]["prompt"] == "<pad>custom"


def test_vllm_unload_shuts_engine_down_then_reloads(vllm_runtime: SimpleNamespace) -> None:
    parser = NemotronParse35(dflash_tokens=None, backend="vllm")
    page = Image.new("RGB", (16, 16))
    parser(page)
    parser.unload()
    vllm_runtime.engine.shutdown.assert_called_once_with(timeout=30.0)
    assert parser._model is None
    parser.unload()
    parser(page)
    assert vllm_runtime.llm.call_count == 2


def test_vllm_shutdown_failure_retains_engine(vllm_runtime: SimpleNamespace) -> None:
    parser = NemotronParse35(dflash_tokens=None, backend="vllm")
    parser(Image.new("RGB", (16, 16)))
    vllm_runtime.engine.shutdown.side_effect = RuntimeError("shutdown failed")
    with pytest.raises(RuntimeError, match="shutdown failed"):
        parser.unload()
    assert parser._model is vllm_runtime.engine


@pytest.mark.parametrize("failure", [False, True])
def test_snapshot_environment_does_not_leak_to_other_engines(
    vllm_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    monkeypatch.delenv(plugin_module.PARSE35_SNAPSHOT_ENV, raising=False)
    if failure:
        vllm_runtime.llm.side_effect = RuntimeError("startup failed")
        with pytest.raises(RuntimeError, match="startup failed"):
            NemotronParse35(dflash_tokens=None, backend="vllm")(Image.new("RGB", (16, 16)))
    else:
        NemotronParse35(dflash_tokens=None, backend="vllm")(Image.new("RGB", (16, 16)))
    assert plugin_module.PARSE35_SNAPSHOT_ENV not in os.environ


@pytest.mark.parametrize("previous_method", [None, "fork", "spawn", "forkserver"])
@pytest.mark.parametrize("previous_snapshot", [None, "/previous/snapshot"])
@pytest.mark.parametrize("failure_stage", [None, "registration", "engine"])
def test_startup_forces_spawn_and_restores_caller_environment(
    vllm_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    previous_method: str | None,
    previous_snapshot: str | None,
    failure_stage: str | None,
) -> None:
    previous_environment = {
        "VLLM_WORKER_MULTIPROC_METHOD": previous_method,
        plugin_module.PARSE35_SNAPSHOT_ENV: previous_snapshot,
    }
    for name, value in previous_environment.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    def check_environment(stage: str) -> None:
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert os.environ[plugin_module.PARSE35_SNAPSHOT_ENV] == vllm_runtime.download.return_value
        if failure_stage == stage:
            raise RuntimeError(f"{stage} failed")

    def register() -> None:
        check_environment("registration")

    def load_engine(**kwargs: object) -> MagicMock:
        check_environment("engine")
        return vllm_runtime.engine

    vllm_runtime.register.side_effect = register
    vllm_runtime.llm.side_effect = load_engine
    parser = NemotronParse35(dflash_tokens=None)
    if failure_stage is None:
        assert parser(Image.new("RGB", (16, 16))) == "<class_Text>page"
    else:
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            parser(Image.new("RGB", (16, 16)))
    for name, value in previous_environment.items():
        assert os.environ.get(name) == value


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_vllm_rejects_unsupported_devices_before_download(device: str) -> None:
    with pytest.raises(ValueError, match="worker GPU set"):
        NemotronParse35(dflash_tokens=None, backend="vllm", device=device)


def test_vllm_cuda_unavailable_fails_before_download(
    vllm_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match='CUDA is unavailable.*device="cpu"'):
        NemotronParse35(dflash_tokens=None, backend="vllm")(Image.new("RGB", (16, 16)))
    vllm_runtime.download.assert_not_called()


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported Parse 3.5 backend"):
        NemotronParse35(dflash_tokens=None, backend="unknown")


def test_cpu_requires_explicit_transformers_backend() -> None:
    with pytest.raises(ValueError, match="requires device"):
        NemotronParse35(dflash_tokens=None, device="cpu")
    assert NemotronParse35(dflash_tokens=None, backend="transformers", device="cpu")._backend == "transformers"


def test_plugin_is_noop_for_other_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(plugin_module.PARSE35_SNAPSHOT_ENV, raising=False)
    plugin_module.register_nemotron_parse35_vllm()


def test_plugin_rejects_unpinned_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(plugin_module.PARSE35_SNAPSHOT_ENV, "/tmp/main")
    with pytest.raises(RuntimeError, match="pinned checkpoint"):
        plugin_module.register_nemotron_parse35_vllm()


def test_plugin_registers_same_snapshot_in_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot = tmp_path / get_hf_revision(NEMOTRON_PARSE35_MODEL)
    package_path = snapshot / "vllm_tied_patch" / "__init__.py"
    package = ModuleType("vllm_tied_patch")
    package.__file__ = str(package_path)
    plugin = ModuleType("vllm_tied_patch.plugin")
    plugin.register = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm_tied_patch", package)
    monkeypatch.setitem(sys.modules, "vllm_tied_patch.plugin", plugin)
    monkeypatch.setenv(plugin_module.PARSE35_SNAPSHOT_ENV, str(snapshot))
    plugin_module.register_nemotron_parse35_vllm()
    plugin.register.assert_called_once()


def test_plugin_loads_package_from_snapshot_without_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / get_hf_revision(NEMOTRON_PARSE35_MODEL)
    package_dir = snapshot / "vllm_tied_patch"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    (package_dir / "plugin.py").write_text(
        "registered = False\ndef register():\n    global registered\n    registered = True\n"
    )
    monkeypatch.delitem(sys.modules, "vllm_tied_patch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm_tied_patch.plugin", raising=False)
    monkeypatch.setenv(plugin_module.PARSE35_SNAPSHOT_ENV, str(snapshot))
    try:
        plugin_module.register_nemotron_parse35_vllm()
        assert sys.modules["vllm_tied_patch.plugin"].registered is True
    finally:
        sys.modules.pop("vllm_tied_patch.plugin", None)
        sys.modules.pop("vllm_tied_patch", None)


def test_plugin_rejects_already_imported_different_version(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("vllm_tied_patch")
    package.__file__ = "/different/vllm_tied_patch/__init__.py"
    monkeypatch.setitem(sys.modules, "vllm_tied_patch", package)
    monkeypatch.setenv(plugin_module.PARSE35_SNAPSHOT_ENV, "/snapshot/" + get_hf_revision(NEMOTRON_PARSE35_MODEL))
    with pytest.raises(RuntimeError, match="different Parse vLLM plugin"):
        plugin_module.register_nemotron_parse35_vllm()


def test_chart_sampling_reaches_local_vllm(vllm_runtime: SimpleNamespace) -> None:
    sampling = dict(temperature=0.1, repetition_penalty=1.0, top_k=50, top_p=1.0, seed=0)
    parser = NemotronParse35(dflash_tokens=None, max_tokens=8744, sampling_params=sampling)
    parser.invoke(Image.new("RGB", (16, 20)))
    assert vllm_runtime.sampling.call_args.kwargs == {
        **sampling,
        "max_tokens": 8744,
        "skip_special_tokens": False,
    }


@pytest.mark.parametrize("tokens", [1, 16, 32])
def test_dflash_uses_drafter_from_same_pinned_snapshot(vllm_runtime, tmp_path, tokens):
    snapshot = tmp_path / get_hf_revision(NEMOTRON_PARSE35_MODEL)
    draft = snapshot / "dflash2"
    draft.mkdir(parents=True, exist_ok=True)
    (draft / "config.json").write_text("{}")
    (draft / "model.safetensors").touch()
    vllm_runtime.download.return_value = str(snapshot)
    parser = NemotronParse35(dflash_tokens=tokens)
    parser.invoke(Image.new("RGB", (16, 16)))
    options = vllm_runtime.llm.call_args.kwargs
    assert options["speculative_config"] == {
        "method": "dflash",
        "model": str(draft),
        "num_speculative_tokens": tokens,
    }
    assert options["hf_overrides"]["architectures"] == ["NRLNemotronParse35DFlashModel"]
    assert vllm_runtime.download.call_args.kwargs["ignore_patterns"] is None


def test_explicit_none_preserves_mtp_without_dflash_drafter(vllm_runtime):
    NemotronParse35(dflash_tokens=None).invoke(Image.new("RGB", (16, 16)))
    assert "speculative_config" not in vllm_runtime.llm.call_args.kwargs
    assert "hf_overrides" not in vllm_runtime.llm.call_args.kwargs
    assert vllm_runtime.download.call_args.kwargs["ignore_patterns"] == ["dflash2/*"]


@pytest.mark.parametrize("missing_asset", ["config.json", "model.safetensors"])
def test_dflash_missing_checkpoint_fails_before_engine_start(vllm_runtime, missing_asset):
    draft = Path(vllm_runtime.download.return_value) / "dflash2"
    (draft / missing_asset).unlink()
    with pytest.raises(RuntimeError, match="missing its DFlash2 checkpoint") as error:
        NemotronParse35(dflash_tokens=32).invoke(Image.new("RGB", (16, 16)))
    message = str(error.value)
    assert str(draft) in message and missing_asset in message
    assert "Refresh the pinned" in message
    assert get_hf_revision(NEMOTRON_PARSE35_MODEL) in message
    assert "dflash_tokens=None" in message and "nemotron_parse_dflash_tokens=None" in message
    assert "MTP fallback" in message
    vllm_runtime.llm.assert_not_called()


@pytest.mark.parametrize("tokens", [0, -1, 33, True, 1.5, "32"])
def test_invalid_dflash_length_fails_before_download(vllm_runtime, tokens):
    with pytest.raises(ValueError, match="draft length"):
        NemotronParse35(dflash_tokens=tokens)
    vllm_runtime.download.assert_not_called()


def test_dflash_rejects_transformers_backend():
    with pytest.raises(ValueError, match="vLLM backend"):
        NemotronParse35(backend="transformers", device="cpu", dflash_tokens=32)


class _Increment(torch.nn.Module):
    def forward(self, values):
        return values + 1


class _TargetBase(torch.nn.Module):
    def _embed_text_input_ids(self, input_ids, embed_input_ids, *, is_multimodal):
        return embed_input_ids(input_ids)

    def forward(self, *args, **kwargs):
        return "base forward"


class _DFlashTarget(plugin_module.Parse35DFlashMixin, _TargetBase):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([_Increment() for _ in range(10)])
        self.decoder.layernorm_embedding = _Increment()
        self.decoder.layer_norm = _Increment()
        self.decoder.embed_tokens = torch.nn.Embedding(4, 2)
        self.decoder.embed_input_ids = self.decoder.embed_tokens
        self.lm_head = torch.nn.Linear(2, 4)


def test_dflash_returns_post_block_states_without_final_normalization():
    model = _DFlashTarget()
    assert model.get_language_model() is model
    assert model.model is model.decoder
    assert model.forward(None, None) == "base forward"
    model.set_aux_hidden_state_layers((2, 5, 9))
    hidden, auxiliary = model.forward(None, None, inputs_embeds=torch.zeros(1, 2))
    torch.testing.assert_close(hidden, torch.full((1, 2), 12.0))
    for actual, expected in zip(auxiliary, [3.0, 6.0, 10.0], strict=True):
        torch.testing.assert_close(actual, torch.full((1, 2), expected))
    token = torch.tensor([1])
    token_hidden, _ = model.forward(token, None)
    torch.testing.assert_close(token_hidden, model.decoder.embed_tokens(token) + 12)


@pytest.mark.parametrize("layers", [(), (0,), (11,), (5, 2), (2, 2)])
def test_dflash_rejects_invalid_auxiliary_layers(layers):
    with pytest.raises(ValueError, match="auxiliary layers"):
        _DFlashTarget().set_aux_hidden_state_layers(layers)


def test_dflash_text_embedding_delegates_to_decoder_without_recursion():
    model = _DFlashTarget()
    ids = torch.tensor([1, 2])

    def recursive_embedding(_):
        pytest.fail("Text embedding delegated back to the multimodal target")

    actual = model._embed_text_input_ids(ids, recursive_embedding, is_multimodal=None)
    torch.testing.assert_close(actual, model.decoder.embed_tokens(ids))


def test_default_vllm_backend_selects_dflash32(vllm_runtime):
    parser = NemotronParse35()
    parser.invoke(Image.new("RGB", (16, 16)))
    assert vllm_runtime.llm.call_args.kwargs["speculative_config"]["num_speculative_tokens"] == 32


def test_transformers_auto_default_does_not_enable_dflash():
    assert NemotronParse35(backend="transformers", device="cpu")._dflash_tokens is None


def _page(value: int = 1) -> np.ndarray:
    return (np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3) * value) % 255


def test_vllm_engine_batches_pages_with_bounded_memory(vllm_runtime: SimpleNamespace) -> None:
    NemotronParse35(dflash_tokens=16)(Image.new("RGB", (16, 16)))
    engine = vllm_runtime.llm.call_args.kwargs
    assert engine["max_num_seqs"] == NemotronParse35().input_batch_size == 128
    assert engine["gpu_memory_utilization"] == 0.3
    assert engine["kv_cache_memory_bytes"] == 24 * 1024**3
    assert engine["mm_processor_cache_gb"] == 0
    assert NemotronParse35(backend="transformers", device="cpu").input_batch_size == 1


def test_vllm_skips_blank_pages_and_keeps_order(vllm_runtime: SimpleNamespace) -> None:
    vllm_runtime.engine.generate.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(text="first", finish_reason="length", token_ids=[1, 2])]),
        SimpleNamespace(outputs=[SimpleNamespace(text="second", finish_reason="stop", token_ids=[3])]),
    ]
    parser = NemotronParse35()
    blank = np.full((16, 16, 3), 255, dtype=np.uint8)
    assert parser.invoke_batch_results([_page(), blank, _page(2)]) == [
        ParsePageResult(text="first", finish_reason="length", output_tokens=2),
        ParsePageResult(text="", finish_reason="stop", output_tokens=0),
        ParsePageResult(text="second", finish_reason="stop", output_tokens=1),
    ]
    assert len(vllm_runtime.engine.generate.call_args.args[0]) == 2


def test_vllm_all_blank_batch_does_not_load_engine(vllm_runtime: SimpleNamespace) -> None:
    assert NemotronParse35().invoke_batch([np.zeros((16, 16, 3), dtype=np.uint8)] * 2) == ["", ""]
    vllm_runtime.llm.assert_not_called()
    vllm_runtime.download.assert_not_called()


def test_concurrent_callers_load_one_engine(vllm_runtime: SimpleNamespace) -> None:
    parser = NemotronParse35()
    with ThreadPoolExecutor(16) as pool:
        assert (
            list(pool.map(lambda _: parser.invoke(Image.new("RGB", (16, 16))), range(32))) == ["<class_Text>page"] * 32
        )
    vllm_runtime.llm.assert_called_once()


@pytest.mark.parametrize("mode", ["normal", "empty", "error", "abort", "error_finish"])
def test_async_engine_keeps_prompt_order_and_propagates_failures(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    preprocess = ModuleType("vllm.renderers.inputs.preprocess")
    preprocess.parse_model_prompt = lambda config, prompt: prompt
    sampling = ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = SimpleNamespace(FINAL_ONLY="final")
    monkeypatch.setitem(sys.modules, preprocess.__name__, preprocess)
    monkeypatch.setitem(sys.modules, sampling.__name__, sampling)
    request_ids = []
    cancelled = []

    async def generate(item: str, params: SimpleNamespace, request_id: str):
        request_ids.append(request_id)
        assert params.output_kind == "final"
        if mode == "error":
            if item == "first":
                raise RuntimeError("request failed")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(item)
                raise
        if mode == "empty":
            return
        await asyncio.sleep(0.01 if item == "first" else 0)
        reason = "error" if mode == "error_finish" and item == "second" else "stop"
        yield SimpleNamespace(
            outputs=[SimpleNamespace(finish_reason="abort" if mode == "abort" else reason)], item=item
        )

    engine = adapter_module._AsyncParseEngine.__new__(adapter_module._AsyncParseEngine)
    engine._loop = asyncio.new_event_loop()
    engine._thread = threading.Thread(target=engine._loop.run_forever)
    engine._thread.start()
    engine._request_ids = itertools.count()
    engine._render_lock = threading.Lock()
    engine.engine = SimpleNamespace(
        model_config=None,
        renderer=SimpleNamespace(
            default_cmpl_tok_params=SimpleNamespace(with_kwargs=lambda **kwargs: kwargs),
            render_cmpl=lambda prompts, params: prompts,
        ),
        generate=generate,
        shutdown=MagicMock(),
    )
    params = SimpleNamespace(clone=SimpleNamespace)
    try:
        if mode == "normal":
            assert [output.item for output in engine.generate(["first", "second"], params)] == ["first", "second"]
            assert len(set(request_ids)) == 2
        else:
            expected = {"empty": "no output", "error": "request failed"}.get(mode, "finished with reason")
            with pytest.raises(RuntimeError, match=expected):
                engine.generate(["first", "second"], params)
            assert cancelled == (["second"] if mode == "error" else [])
        unwound = []

        async def output_handler() -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                unwound.append(True)
                raise

        handler = asyncio.run_coroutine_threadsafe(output_handler(), engine._loop)
        # Like vLLM, engine shutdown only schedules the handler's cancellation on the loop.
        engine.engine.shutdown.side_effect = lambda timeout: handler.cancel()
    finally:
        engine.shutdown(timeout=30)
    engine.engine.shutdown.assert_called_once_with(timeout=30)
    assert unwound == [True]
    assert engine._loop.is_closed()
    assert not engine._thread.is_alive()


def test_vllm_validates_blank_pages(vllm_runtime: SimpleNamespace) -> None:
    with pytest.raises(ValueError):
        NemotronParse35().invoke_batch([np.zeros((16, 16, 2, 2), dtype=np.uint8)])
