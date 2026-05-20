#! python3
"""
Usage:
helm lint nemo_retriever/helm

python ci/scripts/release_helm_chart.py
    -o nvidian
    -t nemo-llm
    -v 26.03
    -n nemo-retriever
    --chart-dir nemo_retriever/helm

Requires: pip install ngcsdk pyyaml
Env vars: NGC_CLI_API_KEY (required for publish)
"""

import argparse
import os
import subprocess
import sys

import yaml

LOGO = "https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/nemo-retriever-graphic.png"


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

    chart = yaml.safe_load(open(f"dist/{n}/Chart.yaml").read())
    chart["name"] = n
    chart["version"] = v
    with open(f"dist/{n}/Chart.yaml", "w") as f:
        f.write(yaml.safe_dump(chart))

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
        print(f"Updating chart metadata for {target} ...")
        clt.registry.chart.update(
            target=target,
            overview_filepath=overview,
            short_description=d,
            logo=logo,
            display_name=dn,
            publisher="NVIDIA",
        )

        print(f"Pushing chart {target}:{v} ...")
        clt.registry.chart.push(
            target=f"{target}:{v}",
            source_dir=".",
        )
        print(f"Successfully pushed {target}:{v}")


if __name__ == "__main__":
    main()
