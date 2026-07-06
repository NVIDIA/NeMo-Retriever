# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import get_args

import nemo_retriever.common.params as params_module
from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.params import IngestorRunMode


def test_run_mode_type_aliases_are_domain_specific() -> None:
    assert set(get_args(IngestorRunMode)) == {"inprocess", "batch", "service"}


def test_generic_run_mode_aliases_are_not_exported() -> None:
    params_models = import_module("nemo_retriever.common.params.models")
    model_module = import_module("nemo_retriever.models.model")

    assert not hasattr(params_module, "RunMode")
    assert "RunMode" not in params_module.__all__
    assert not hasattr(params_models, "RunMode")
    assert not hasattr(model_module, "RunMode")


def test_model_runmode_alias_is_removed() -> None:
    """``ModelRunMode`` was dead metadata (only ever ``"local"``) and has been removed."""
    model_module = import_module("nemo_retriever.models.model")

    assert not hasattr(model_module, "ModelRunMode")


def test_fused_mode_tuning_surface_is_not_exported() -> None:
    params_models = import_module("nemo_retriever.common.params.models")

    assert not hasattr(params_module, "FusedTuningParams")
    assert "FusedTuningParams" not in params_module.__all__
    assert not hasattr(params_models, "FusedTuningParams")
    assert "fused_tuning" not in EmbedParams.model_fields
