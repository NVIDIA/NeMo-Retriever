# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``wrap_callable_as_stage`` (kept out of ``.../util/pipeline/`` so Ray/cloudpickle
does not mis-resolve the module as ``util.pipeline`` on workers)."""

import pandas as pd
import pytest
import ray
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.util.pipeline.tools import (
    wrap_callable_as_stage,
)
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


@pytest.fixture(scope="session", autouse=True)
def ray_startup_and_shutdown():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


class DummyConfig(BaseModel):
    foo: int
    bar: str = "baz"


def _stage_fn_set_bar(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
    control_message.set_metadata("bar", stage_config.bar)
    return control_message


def _stage_fn_set_result(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
    control_message.set_metadata("result", stage_config.foo)
    return control_message


def _stage_fn_raise(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
    raise RuntimeError("fail!")


def _stage_fn_chain_foo(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
    control_message.set_metadata("foo", stage_config.foo)
    return control_message


def _stage_fn_chain_bar(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
    control_message.set_metadata("bar", stage_config.bar)
    return control_message


def test_stage_processes_message_with_model_config(ray_startup_and_shutdown):
    Stage = wrap_callable_as_stage(_stage_fn_set_bar, DummyConfig)
    cfg = DummyConfig(foo=5, bar="quux")
    stage = Stage.remote(config=cfg)
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out = ray.get(stage.on_data.remote(message))
    assert out.get_metadata("bar") == "quux"


def test_stage_processes_message_with_dict_config(ray_startup_and_shutdown):
    """Given a valid config dict, on_data returns the value from the wrapped function."""
    Stage = wrap_callable_as_stage(_stage_fn_set_result, DummyConfig)
    cfg = {"foo": 7}
    stage = Stage.remote(config=cfg)
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out = ray.get(stage.on_data.remote(message))

    assert out is not None
    assert out.get_metadata("result") == 7


def test_stage_returns_original_message_on_error(ray_startup_and_shutdown):
    Stage = wrap_callable_as_stage(_stage_fn_raise, DummyConfig)
    stage = Stage.remote(config={"foo": 1})
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out = ray.get(stage.on_data.remote(message))
    assert isinstance(out, IngestControlMessage)


def test_stage_can_chain_calls(ray_startup_and_shutdown):
    Stage1 = wrap_callable_as_stage(_stage_fn_chain_foo, DummyConfig)
    Stage2 = wrap_callable_as_stage(_stage_fn_chain_bar, DummyConfig)
    stage1 = Stage1.remote(config={"foo": 42})
    stage2 = Stage2.remote(config={"foo": 0, "bar": "chained!"})
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out1 = ray.get(stage1.on_data.remote(message))
    out2 = ray.get(stage2.on_data.remote(out1))

    assert out1.get_metadata("foo") == 42
    assert out2.get_metadata("bar") == "chained!"
