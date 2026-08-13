#! python3
"""
Usage:
helm lint nemo_retriever/helm

python ci/scripts/release_helm_chart.py
    -o <ngc-org> -t <ngc-team> -v <chart-version> -n nemo-retriever \\
    --chart-dir nemo_retriever/helm

Requires: pip install ngcsdk pyyaml
Env vars: NGC_CLI_API_KEY (required for publish). In CI, org/team come from
NGC_ORG and NGC_TEAM repository secrets (not committed to the repo).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode

LOGO = "https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/nemo-retriever-graphic.png"

_NOT_FOUND_EXC = frozenset({"ResourceNotFoundException", "ChartNotFoundException"})
_ALREADY_EXISTS_EXC = frozenset({"ResourceAlreadyExistsException", "ChartAlreadyExistsException"})

# Tags that move over time. A published release must not depend on them because
# the artifact a customer pulls would drift away from the qualified set.
_FLOATING_TAGS = frozenset({"latest", "main", "master", "dev", "devel", "nightly", "edge", "stable"})


def _exc_name(exc: BaseException) -> str:
    return type(exc).__name__


def _set_yaml_scalar(path: Path, key_path: tuple[str, ...], value: str) -> None:
    """Update one YAML scalar while preserving comments and formatting."""
    source = path.read_text(encoding="utf-8")
    node = yaml.compose(source)
    if not isinstance(node, MappingNode):
        raise ValueError(f"{path} must contain a YAML mapping")

    current = node
    for key in key_path:
        if not isinstance(current, MappingNode):
            raise KeyError(f"{'.'.join(key_path)} is not a mapping path in {path}")
        match = next(
            (child for key_node, child in current.value if isinstance(key_node, ScalarNode) and key_node.value == key),
            None,
        )
        if match is None:
            raise KeyError(f"{'.'.join(key_path)} is missing from {path}")
        current = match

    if not isinstance(current, ScalarNode):
        raise ValueError(f"{'.'.join(key_path)} must be a scalar in {path}")

    replacement = json.dumps(value)
    updated = source[: current.start_mark.index] + replacement + source[current.end_mark.index :]
    path.write_text(updated, encoding="utf-8")


def _set_release_versions(chart_dir: Path, name: str, version: str) -> None:
    chart_path = chart_dir / "Chart.yaml"
    _set_yaml_scalar(chart_path, ("name",), name)
    _set_yaml_scalar(chart_path, ("version",), version)
    _set_yaml_scalar(chart_path, ("appVersion",), version)
    _set_yaml_scalar(chart_dir / "values.yaml", ("service", "image", "tag"), version)


def _collect_image_references(values: object, prefix: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    """Return every ``(values path, repository, tag)`` image the chart deploys.

    Blocks whose ``repository`` is empty are skipped because the chart resolves
    those to another image (for example ``service.gpuImage``).
    """
    if not isinstance(values, dict):
        return []

    references: list[tuple[str, str, str]] = []
    repository = values.get("repository")
    if isinstance(repository, str) and repository.strip():
        tag = values.get("tag")
        references.append((".".join(prefix), repository.strip(), str(tag if tag is not None else "").strip()))

    for key, child in values.items():
        references.extend(_collect_image_references(child, prefix + (str(key),)))

    return references


def _validate_pinned_image_tags(references: list[tuple[str, str, str]]) -> None:
    offenders = [
        f"{path}: {repository}:{tag or '<empty>'}"
        for path, repository, tag in references
        if not tag or tag.lower() in _FLOATING_TAGS
    ]
    if offenders:
        raise ValueError(
            "Every image in a published chart must pin an explicit, non-floating tag. Offending values:\n  "
            + "\n  ".join(offenders)
        )


def _render_image_inventory(references: list[tuple[str, str, str]], name: str, version: str) -> str:
    """Render the artifact set a chart release deploys, for release qualification."""
    lines = [
        f"### Helm chart artifact inventory - `{name}:{version}`",
        "",
        "Compare this packaged artifact set against the release test plan matrix.",
        "",
        "| Values path | Image |",
        "|-------------|-------|",
    ]
    lines += [f"| `{path}` | `{repository}:{tag}` |" for path, repository, tag in sorted(references)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Release helm chart to specified org and team.")
    parser.add_argument(
        "-o",
        "--org",
        action="store",
        help="The target ngc org to deploy to",
        required=True,
    )
    parser.add_argument(
        "--target-org",
        action="store",
        help="The target ngc org to reference in docs",
    )
    parser.add_argument(
        "-t",
        "--team",
        action="store",
        help="The target ngc team to deploy to",
        required=True,
    )
    parser.add_argument(
        "--target-team",
        action="store",
        help="The target ngc team to reference in docs",
    )
    parser.add_argument(
        "-n",
        "--name",
        action="store",
        help="The name of the chart",
        required=True,
    )
    parser.add_argument(
        "--chart-dir",
        action="store",
        help="Path to the Helm chart source directory",
        default="nemo_retriever/helm",
    )
    parser.add_argument(
        "--display-name",
        action="store",
        help="The display name of the chart",
        default="NVIDIA NeMo Retriever Helm Chart",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store",
        help="The version of the chart",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--description",
        action="store",
        help="The description of the chart",
        default="Helm chart for the NeMo Retriever ingest service and optional NIM microservices",
    )
    parser.add_argument(
        "-l",
        "--logo-url",
        action="store",
        help="The logo of the chart",
    )

    parser.add_argument("-r", "--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--image-inventory",
        action="store",
        help="Write a Markdown inventory of every image the packaged chart deploys to this path",
    )
    args = parser.parse_args()

    n = args.name
    o = args.org
    t = args.team
    v = args.version
    d = args.description
    dn = args.display_name
    chart_dir = args.chart_dir

    if not os.path.isdir(chart_dir):
        print(f"ERROR: chart directory does not exist: {chart_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(f"dist/{n}", exist_ok=True)
    subprocess.check_call(
        f"""
    rm -rf dist/{n}/*
    cp -r {chart_dir}/* dist/{n}/
    echo $(git rev-parse --short HEAD) >> dist/{n}/.gitsha
    """,
        shell=True,
    )

    _set_release_versions(Path("dist") / n, n, v)

    packaged_values = yaml.safe_load((Path("dist") / n / "values.yaml").read_text(encoding="utf-8"))
    image_references = _collect_image_references(packaged_values)
    _validate_pinned_image_tags(image_references)
    inventory = _render_image_inventory(image_references, n, v)
    print(inventory)
    if args.image_inventory:
        Path(args.image_inventory).write_text(inventory, encoding="utf-8")

    overview = f"dist/{n}/README.md"
    logo = args.logo_url if args.logo_url else LOGO

    subprocess.check_call(f"helm package dist/{n}", shell=True)

    if args.dry_run:
        print(f"[DRY RUN] Chart packaged successfully: {n}-{v}.tgz")
        print(f"[DRY RUN] Skipping NGC chart update and push for {o}/{t}/{n}:{v}")
    else:
        api_key = os.environ.get("NGC_CLI_API_KEY", "")
        if not api_key:
            print("ERROR: NGC_CLI_API_KEY environment variable is not set", file=sys.stderr)
            sys.exit(1)

        from ngcsdk import Client

        clt = Client()
        clt.configure(api_key=api_key, org_name=o, team_name=t)

        target = f"{o}/{t}/{n}"
        metadata_kwargs = dict(
            overview_filepath=overview,
            short_description=d,
            logo=logo,
            display_name=dn,
            publisher="NVIDIA",
        )
        print(f"Updating chart metadata for {target} ...")
        try:
            clt.registry.chart.update(target=target, **metadata_kwargs)
        except Exception as exc:
            if _exc_name(exc) not in _NOT_FOUND_EXC:
                raise
            print(f"Chart '{target}' not found ({_exc_name(exc)}); creating registry entry ...")
            clt.registry.chart.create(target=target, **metadata_kwargs)

        print(f"Pushing chart {target}:{v} ...")
        try:
            clt.registry.chart.push(
                target=f"{target}:{v}",
                source_dir=".",
            )
            print(f"Successfully pushed {target}:{v}")
        except Exception as exc:
            if _exc_name(exc) not in _ALREADY_EXISTS_EXC:
                raise
            print(
                f"Chart version '{v}' already exists in NGC ({_exc_name(exc)}); "
                "skipping push. Re-run with a new version tag to publish different chart contents."
            )


if __name__ == "__main__":
    main()
