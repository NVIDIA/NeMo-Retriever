from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    return str(value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dataframe_sample(df: Any, limit: int = 3) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.head(limit).to_dict(orient="records")


def ray_state(ray_module: Any) -> dict[str, Any]:
    nodes = []
    for node in ray_module.nodes():
        nodes.append(
            {
                "node_id": node.get("NodeID"),
                "alive": bool(node.get("Alive")),
                "node_manager_address": node.get("NodeManagerAddress"),
                "resources": node.get("Resources", {}),
            }
        )

    return {
        "cluster_resources": ray_module.cluster_resources(),
        "available_resources": ray_module.available_resources(),
        "nodes": nodes,
    }


def alive_node_count(state: dict[str, Any]) -> int:
    return sum(1 for node in state.get("nodes", []) if node.get("alive"))


def assert_cluster_ready(state: dict[str, Any], expected_nodes: int = 2) -> None:
    alive_nodes = alive_node_count(state)
    if alive_nodes < expected_nodes:
        raise AssertionError(f"expected at least {expected_nodes} alive Ray nodes, found {alive_nodes}")

    resources = state.get("cluster_resources", {})
    for resource_name in ("nemo_head", "nemo_worker"):
        resource_value = resources.get(resource_name, 0)
        if resource_value < 1:
            raise AssertionError(f"expected Ray cluster resource {resource_name!r} >= 1, found {resource_value!r}")
