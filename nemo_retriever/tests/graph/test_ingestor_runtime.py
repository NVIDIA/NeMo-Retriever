import pytest

from nemo_retriever.common.params import EmbedParams, ExtractParams
from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources
from nemo_retriever.graph.ingestor_runtime import batch_tuning_to_node_overrides, build_graph
from nemo_retriever.graph.operator_resolution import resolve_graph


def _cluster(cpu_count: int) -> ClusterResources:
    resources = Resources(cpu_count=cpu_count, gpu_count=1)
    return ClusterResources(total_resources=resources, available_resources=resources)


def test_batch_tuning_to_node_overrides_parallelizes_text_extraction_on_cpu() -> None:
    overrides = batch_tuning_to_node_overrides(
        extract_params=ExtractParams(),
        embed_params=EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"),
        cluster_resources=_cluster(32),
        extraction_mode="text",
    )

    assert overrides["MultiTypeExtractOperator"] == {
        "concurrency": 8,
        "num_cpus": 1,
        "num_gpus": 0.0,
    }


@pytest.mark.parametrize(("cpu_count", "expected_concurrency"), [(1, 1), (8, 2), (32, 8), (224, 8)])
def test_batch_tuning_to_node_overrides_bounds_text_extraction_pool(cpu_count: int, expected_concurrency: int) -> None:
    overrides = batch_tuning_to_node_overrides(
        extract_params=ExtractParams(),
        embed_params=None,
        cluster_resources=_cluster(cpu_count),
        extraction_mode="text",
    )

    assert overrides["MultiTypeExtractOperator"]["concurrency"] == expected_concurrency


def test_batch_tuning_to_node_overrides_leaves_non_text_multi_type_actor_unchanged() -> None:
    overrides = batch_tuning_to_node_overrides(
        extract_params=ExtractParams(),
        embed_params=None,
        cluster_resources=_cluster(32),
        extraction_mode="auto",
    )

    assert "MultiTypeExtractOperator" not in overrides


def test_text_extraction_override_targets_preserved_resolved_graph_node_name() -> None:
    cluster = _cluster(32)
    graph = build_graph(extraction_mode="text", extract_params=ExtractParams(), stage_order=("extract",))
    resolved_graph = resolve_graph(graph, cluster)
    overrides = batch_tuning_to_node_overrides(
        extract_params=ExtractParams(),
        embed_params=None,
        cluster_resources=cluster,
        extraction_mode="text",
    )

    assert resolved_graph.roots[0].name == "MultiTypeExtractOperator"
    assert resolved_graph.roots[0].operator_class.__name__ == "MultiTypeExtractCPUActor"
    assert resolved_graph.roots[0].name in overrides
    assert "MultiTypeExtractCPUActor" not in overrides
