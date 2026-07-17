# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nemo_retriever.harness.benchmark_registry import (
    get_benchmark,
    get_dataset,
    get_runset,
    VIDORE_V3_EMBED_MODEL,
    VIDORE_V3_PUBLIC_DATASETS,
)
from nemo_retriever.harness.resolution import resolve_benchmark
from nemo_retriever.harness.runfile import load_runfile


VIDORE_V3_DATASET_FACTS = {
    "vidore_v3_computer_science": (2, 1360, 1290),
    "vidore_v3_energy": (41, 2225, 1848),
    "vidore_v3_finance_en": (6, 2942, 1854),
    "vidore_v3_finance_fr": (5, 2384, 1920),
    "vidore_v3_hr": (14, 1110, 1908),
    "vidore_v3_industrial": (27, 5244, 1698),
    "vidore_v3_pharmaceuticals": (52, 2313, 2184),
    "vidore_v3_physics": (42, 1674, 1812),
}
RUNFILES_DIR = Path(__file__).resolve().parents[1] / "harness" / "runfiles"


def test_checked_in_runfiles_leave_resource_tuning_automatic() -> None:
    runfiles = sorted(RUNFILES_DIR.glob("*.json"))
    explicit_tuning: dict[str, list[str]] = {}
    for path in runfiles:
        request = load_runfile(path)
        resolved = resolve_benchmark(
            request.benchmark,
            mode=request.mode or "local",
            overrides=request.overrides,
        )
        resource_overrides = [override.split("=", 1)[0] for override in request.overrides if ".batch." in override]
        configured_batches = [
            f"ingest.{stage}.batch"
            for stage, config in resolved["ingest"].items()
            if isinstance(config, dict) and config.get("batch")
        ]
        explicit_tuning[path.name] = [*resource_overrides, *configured_batches]

    assert runfiles
    assert explicit_tuning == {path.name: [] for path in runfiles}


@pytest.mark.parametrize("dataset_name", VIDORE_V3_DATASET_FACTS)
def test_vidore_v3_benchmarks_use_page_level_multimodal_defaults(dataset_name: str) -> None:
    dataset = get_dataset(dataset_name)
    benchmark = get_benchmark(f"{dataset_name}_beir")

    assert dataset.beir_loader == "vidore_hf"
    assert dataset.beir_doc_id_field == "pdf_page"
    assert Path(dataset.path).name == dataset_name
    assert benchmark.dataset == dataset_name
    assert benchmark.ingest["extract"] == {
        "extract_infographics": True,
        "extract_page_as_image": True,
    }
    assert benchmark.ingest["embed"] == {
        "embed_model_name": VIDORE_V3_EMBED_MODEL,
        "embed_modality": "text_image",
        "embed_granularity": "page",
    }
    assert benchmark.query["embed_model_name"] == VIDORE_V3_EMBED_MODEL
    assert benchmark.evaluation["dataset_name"] == dataset_name
    assert benchmark.evaluation["doc_id_field"] == "pdf_page"
    assert ("finance" in benchmark.tags) is ("finance" in VIDORE_V3_PUBLIC_DATASETS[dataset_name].lower())


@pytest.mark.parametrize(("dataset_name", "facts"), VIDORE_V3_DATASET_FACTS.items())
def test_vidore_v3_runfiles_carry_dataset_integrity_gates(dataset_name: str, facts: tuple[int, int, int]) -> None:
    files, pages, queries = facts
    request = load_runfile(RUNFILES_DIR / f"{dataset_name}_beir.json")

    assert request.benchmark == f"{dataset_name}_beir"
    assert request.mode == "batch"
    assert request.requirements == (
        f"files=={files}",
        f"pages=={pages}",
        f"query_count=={queries}",
    )


def test_vidore_v3_all_runset_contains_every_public_dataset() -> None:
    runset = get_runset("vidore_v3_all")

    assert runset.runs == tuple(f"{name}_beir" for name in VIDORE_V3_PUBLIC_DATASETS)
