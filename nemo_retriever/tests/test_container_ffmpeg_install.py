# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest import SkipTest, TestCase, main


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


class ContainerFfmpegInstallTests(TestCase):
    def test_dockerfile_policy_test_skips_when_repo_root_not_available(self) -> None:
        missing_dockerfile = Path("/tmp/nemo-retriever-missing-root/Dockerfile")

        with self.assertRaises(SkipTest):
            _read_required_file(missing_dockerfile)

    def test_dockerfile_uses_runtime_ffmpeg_install_without_build_arg(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        dockerfile = _read_required_file(repo_root / "Dockerfile")

        self.assertNotIn("ARG INSTALL_FFMPEG", dockerfile)
        self.assertNotIn("--build-arg INSTALL_FFMPEG=true", dockerfile)
        self.assertNotIn('RUN if [ "${INSTALL_FFMPEG}" = "true" ]', dockerfile)
        self.assertNotIn("docker/scripts/install_ffmpeg.sh", dockerfile)
        self.assertNotIn("ffmpeg.org/releases", dockerfile)

    def test_service_image_can_install_ffmpeg_at_runtime_with_limited_sudo(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        dockerfile = _read_required_file(repo_root / "Dockerfile")

        self.assertIn("      sudo \\", dockerfile)
        self.assertIn(
            "COPY docker/scripts/retriever_service_entrypoint.sh /usr/local/bin/retriever-service-entrypoint",
            dockerfile,
        )
        self.assertIn(
            "COPY docker/scripts/retriever_install_ffmpeg.sh /usr/local/sbin/retriever-install-ffmpeg",
            dockerfile,
        )
        self.assertIn('ENTRYPOINT ["/usr/local/bin/retriever-service-entrypoint"]', dockerfile)
        self.assertIn("nemo ALL=(root) NOPASSWD: /usr/local/sbin/retriever-install-ffmpeg", dockerfile)
        self.assertNotIn("NOPASSWD: /usr/bin/apt-get update", dockerfile)
        self.assertNotIn("NOPASSWD: /usr/bin/apt-get install", dockerfile)
        self.assertNotIn("NOPASSWD: /usr/bin/apt-get clean", dockerfile)

    def test_service_entrypoint_installs_ffmpeg_when_runtime_flag_enabled(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        _read_required_file(repo_root / "Dockerfile")
        entrypoint_path = repo_root / "docker/scripts/retriever_service_entrypoint.sh"

        self.assertTrue(entrypoint_path.is_file(), f"service entrypoint not present: {entrypoint_path}")
        entrypoint = entrypoint_path.read_text(encoding="utf-8")

        self.assertIn("INSTALL_FFMPEG:-false", entrypoint)
        self.assertIn("command -v ffmpeg", entrypoint)
        self.assertIn("command -v ffprobe", entrypoint)
        self.assertIn("sudo /usr/local/sbin/retriever-install-ffmpeg", entrypoint)
        self.assertNotIn("sudo apt-get update", entrypoint)
        self.assertNotIn("sudo apt-get install", entrypoint)
        self.assertIn('exec "$@"', entrypoint)

    def test_runtime_ffmpeg_installer_rejects_arguments(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        installer_path = repo_root / "docker/scripts/retriever_install_ffmpeg.sh"

        installer = _read_required_file(installer_path)

        self.assertIn('if [ "$#" -ne 0 ]', installer)
        self.assertIn("/usr/bin/apt-get update", installer)
        self.assertIn("/usr/bin/apt-get install -y --no-install-recommends ffmpeg", installer)
        self.assertIn("/usr/bin/apt-get clean", installer)
        self.assertNotIn("sudo ", installer)

    def test_helm_chart_exposes_first_class_runtime_ffmpeg_value(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        values = _read_required_file(repo_root / "nemo_retriever/helm/values.yaml")
        deployment = _read_required_file(repo_root / "nemo_retriever/helm/templates/deployment.yaml")

        self.assertIn("installFfmpeg: false", values)
        self.assertIn("service.installFfmpeg", values)
        self.assertIn("cannot both set INSTALL_FFMPEG", deployment)
        self.assertEqual(deployment.count("- name: INSTALL_FFMPEG"), 2)
        self.assertEqual(deployment.count("{{- if $svc.installFfmpeg }}"), 2)

    def test_helm_docs_describe_runtime_ffmpeg_caveats(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        helm_readme = _read_required_file(repo_root / "nemo_retriever/helm/README.md")

        self.assertIn("service.installFfmpeg", helm_readme)
        self.assertIn("INSTALL_FFMPEG=true", helm_readme)
        self.assertIn("allowPrivilegeEscalation: false", helm_readme)
        self.assertIn("readOnlyRootFilesystem: true", helm_readme)
        self.assertIn("network egress", helm_readme)

    def test_source_docs_do_not_document_ffmpeg_build_arg(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        docs = (
            repo_root / "nemo_retriever/README.md",
            repo_root / "nemo_retriever/helm/README.md",
            repo_root / "docs/docs/extraction/audio-video.md",
            repo_root / "docs/docs/extraction/deployment-options.md",
            repo_root / "docs/docs/extraction/prerequisites-support-matrix.md",
            repo_root / "docs/docs/extraction/releasenotes.md",
            repo_root / "docs/docs/extraction/troubleshoot.md",
        )

        for path in docs:
            with self.subTest(path=path):
                text = _read_required_file(path)
                self.assertNotIn("--build-arg INSTALL_FFMPEG=true", text)
                self.assertNotIn("build an ffmpeg-enabled", text)

    def test_deployment_options_describes_runtime_ffmpeg_install_for_helm(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        deployment_options = _read_required_file(repo_root / "docs/docs/extraction/deployment-options.md")

        self.assertIn("service.installFfmpeg=true", deployment_options)
        self.assertIn("runtime", deployment_options)
        self.assertNotIn("must run a service image that already includes", deployment_options)
        self.assertNotIn("does not install operating system packages", deployment_options)


if __name__ == "__main__":
    main()
