# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision a Helm service around one portable harness run-files session."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence

from nemo_retriever.harness.helm_config import HelmDeploymentConfig, load_helm_config
from nemo_retriever.harness.helm_manager import HelmServiceManager

logger = logging.getLogger(__name__)

EXIT_HELM_FAILURE = 4


def build_run_files_command(
    config: HelmDeploymentConfig,
    runfiles: Sequence[Path],
    *,
    output_dir: Path,
    session_name: str,
    dataset_paths: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nemo_retriever.harness",
        "run-files",
        "--output-dir",
        str(output_dir),
        "--session-name",
        session_name,
        "--mode",
        "service",
        "--service-endpoint",
        f"http://localhost:{config.helm_service_local_port}",
    ]
    if dataset_paths is not None:
        command.extend(("--dataset-paths", str(dataset_paths)))
    command.extend(str(path) for path in runfiles)
    return command


def run_helm_session(
    config_path: Path,
    runfiles: Sequence[Path],
    *,
    output_dir: Path,
    session_name: str = "helm_service",
    dataset_paths: Path | None = None,
    manager_factory: Callable[[HelmDeploymentConfig], HelmServiceManager] = HelmServiceManager,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Deploy, run the shared harness session, collect failure logs, and tear down."""
    config = load_helm_config(config_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = manager_factory(config)
    exit_code = EXIT_HELM_FAILURE
    collect_logs = False
    try:
        if manager.start() != 0:
            logger.error("Managed Helm service failed to become ready.")
            collect_logs = True
        else:
            command = build_run_files_command(
                config,
                runfiles,
                output_dir=output_dir,
                session_name=session_name,
                dataset_paths=dataset_paths,
            )
            completed = command_runner(command, check=False)
            exit_code = int(completed.returncode)
            collect_logs = exit_code != 0
    except Exception:
        collect_logs = True
        logger.exception("Managed Helm harness session failed.")
        exit_code = EXIT_HELM_FAILURE
    finally:
        if collect_logs:
            try:
                manager.dump_logs(output_dir)
            except Exception:
                logger.exception("Could not collect Helm service logs.")
        try:
            teardown_rc = manager.stop(uninstall=True)
        except Exception:
            teardown_rc = EXIT_HELM_FAILURE
            logger.exception("Could not tear down the managed Helm service.")
        if teardown_rc != 0:
            try:
                manager.dump_logs(output_dir)
            except Exception:
                logger.exception("Could not collect logs after Helm teardown failure.")
            if exit_code == 0:
                exit_code = EXIT_HELM_FAILURE
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runfiles", nargs="+", type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Non-secret Helm deployment YAML.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Portable run-files session directory.")
    parser.add_argument("--session-name", default="helm_service")
    parser.add_argument("--dataset-paths", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    raise SystemExit(
        run_helm_session(
            args.config,
            args.runfiles,
            output_dir=args.output_dir,
            session_name=args.session_name,
            dataset_paths=args.dataset_paths,
        )
    )


if __name__ == "__main__":
    main()
