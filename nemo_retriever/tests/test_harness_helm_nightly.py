# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo_retriever.harness.artifact_writer import ArtifactWriter, artifact_paths
from nemo_retriever.harness.beir_runner import run_service_beir_queries
from nemo_retriever.harness.contracts import EXIT_INGEST_FAILURE, EXIT_METRIC_GATE_FAILURE, HarnessRunError
from nemo_retriever.harness.execution import PreparedBenchmark, preflight_benchmark, run_prepared_benchmark
from nemo_retriever.harness.helm_config import load_helm_config
from nemo_retriever.harness.helm_manager import HelmServiceManager
from nemo_retriever.harness.helm_runner import EXIT_HELM_FAILURE, run_helm_session
from nemo_retriever.harness.resolution import build_service_query_request
from nemo_retriever.query.options import QueryServiceOptions, ServiceQueryRequest
from nemo_retriever.tools.recall.beir import BeirDataset


def _write_helm_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "helm_chart: chart",
                "service_image_repository: registry.example/nrl-service",
                "service_image_tag: main-abc123",
                "helm_release: helm-test",
                "helm_namespace: helm-test",
                "helm_service_local_port: 17670",
            )
        ),
        encoding="utf-8",
    )


def _prepared(tmp_path: Path, *, dry_run: bool = False, requirements=(), evaluation_mode: str = "none"):
    ingest_request = SimpleNamespace(
        documents=[str(tmp_path / "one.txt")],
        input_type="txt",
        connection=SimpleNamespace(service_url="http://localhost:17670", service_concurrency=8),
    )
    query_request = SimpleNamespace(
        retrieval=SimpleNamespace(top_k=5, candidate_k=None, page_dedup=False, content_types=None)
    )
    return PreparedBenchmark(
        benchmark="jp20_beir",
        mode="service",
        overrides=(),
        requirements=tuple(requirements),
        dry_run=dry_run,
        resolved={
            "dataset": {"name": "jp20"},
            "ingest": {"run_mode": "service"},
            "evaluation": {"mode": evaluation_mode},
            "summary_keys": ["files", "rows_processed", "query_count", "recall_5"],
        },
        dataset_path=tmp_path,
        service_endpoint="http://localhost:17670",
        service_ingest_request=ingest_request,
        service_query_request=query_request,
    )


@pytest.fixture
def service_execution(monkeypatch):
    import nemo_retriever.harness.execution as execution

    monkeypatch.setattr(
        execution,
        "service_plan_payload",
        lambda ingest, query: {
            "service_url": ingest.connection.service_url,
            "documents": list(ingest.documents),
            "query": {"top_k": query.retrieval.top_k},
        },
    )
    return execution


def test_service_preflight_builds_product_requests(monkeypatch, tmp_path: Path) -> None:
    import nemo_retriever.harness.execution as execution

    resolved = {"dataset": {"name": "jp20"}, "ingest": {}, "query": {}, "evaluation": {}}
    plan = object()
    ingest = object()
    query = object()
    calls = []
    monkeypatch.setattr(execution, "resolve_benchmark", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(execution, "validate_dataset_inputs", lambda *_args, **_kwargs: (tmp_path, None))
    monkeypatch.setattr(
        execution,
        "build_service_ingest_plan_request",
        lambda *_args, **kwargs: calls.append(("ingest", kwargs["service_url"])) or plan,
    )
    monkeypatch.setattr(execution, "resolve_service_ingest_request", lambda request: ingest)
    monkeypatch.setattr(
        execution,
        "build_service_query_request",
        lambda *_args, **kwargs: calls.append(("query", kwargs["service_url"])) or query,
    )

    prepared = preflight_benchmark(
        "jp20_beir",
        mode="service",
        overrides=(),
        requirements=(),
        dry_run=True,
        service_endpoint="http://localhost:17670",
    )

    assert prepared.service_ingest_request is ingest
    assert prepared.service_query_request is query
    assert calls == [("ingest", "http://localhost:17670"), ("query", "http://localhost:17670")]


def test_service_dry_run_writes_plans_without_network(service_execution, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        service_execution,
        "execute_service_ingest_request",
        lambda _request: pytest.fail("dry-run contacted the service"),
    )
    outcome = run_prepared_benchmark(_prepared(tmp_path, dry_run=True), output_dir=str(tmp_path / "artifacts"))

    assert outcome.exit_code == 0
    assert outcome.results["mode"] == "service"
    assert outcome.results["summary_metrics"]["files"] == 1
    assert not (outcome.artifact_dir / "summary_metrics.json").exists()


def test_service_ingest_and_beir_use_shared_results(service_execution, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        service_execution,
        "execute_service_ingest_request",
        lambda _request: SimpleNamespace(to_summary_dict=lambda: {"n_rows": 12}),
    )
    monkeypatch.setattr(
        service_execution,
        "run_service_beir_queries",
        lambda *_args: ([10.0], {"Recall@5": 0.9}, 1),
    )

    outcome = run_prepared_benchmark(
        _prepared(tmp_path, evaluation_mode="beir", requirements=("recall_5>=0.85",)),
        output_dir=str(tmp_path / "artifacts"),
    )

    assert outcome.exit_code == 0
    assert outcome.results["summary_metrics"]["rows_processed"] == 12
    assert outcome.results["summary_metrics"]["query_count"] == 1
    assert outcome.results["summary_metrics"]["recall_5"] == 0.9


def test_service_beir_queries_each_dataset_query(monkeypatch, tmp_path: Path) -> None:
    import nemo_retriever.harness.beir_runner as beir_runner

    dataset = BeirDataset(
        dataset_name="test",
        query_ids=["q1", "q2"],
        queries=["first", "second"],
        qrels={"q1": {"one.pdf": 1}, "q2": {"two.pdf": 1}},
    )
    seen_queries = []
    monkeypatch.setattr(beir_runner, "load_beir_dataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(
        beir_runner,
        "query_service_documents",
        lambda request: seen_queries.append(request.query) or [{"pdf_basename": f"{request.query}.pdf", "_score": 1.0}],
    )
    monkeypatch.setattr(
        beir_runner,
        "build_beir_run_from_hits",
        lambda query_ids, hits, **_kwargs: {
            query_id: {hit[0]["pdf_basename"]: 1.0} for query_id, hit in zip(query_ids, hits)
        },
    )
    monkeypatch.setattr(beir_runner, "compute_beir_metrics", lambda *_args, **_kwargs: {"Recall@5": 1.0})
    writer = ArtifactWriter(artifact_dir=tmp_path, run_id="run", benchmark="jp20_beir")

    latencies, metrics, count = run_service_beir_queries(
        writer,
        {"evaluation": {"loader": "jp20_csv", "dataset_name": "test"}},
        ServiceQueryRequest(query="", service=QueryServiceOptions(service_url="http://localhost:17670")),
    )

    assert seen_queries == ["first", "second"]
    assert len(latencies) == 2
    assert metrics == {"Recall@5": 1.0}
    assert count == 2
    assert len(writer.path("query_results.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_service_metric_gate_failure_uses_standard_exit_code(service_execution, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        service_execution,
        "execute_service_ingest_request",
        lambda _request: SimpleNamespace(to_summary_dict=lambda: {"n_rows": 1}),
    )
    outcome = run_prepared_benchmark(
        _prepared(tmp_path, requirements=("files==2",)),
        output_dir=str(tmp_path / "artifacts"),
    )

    assert outcome.exit_code == EXIT_METRIC_GATE_FAILURE
    assert outcome.results["failure"]["failure_reason"] == "metric_gate_failed"


def test_service_ingest_failure_is_concise(service_execution, monkeypatch, tmp_path: Path) -> None:
    def fail(_request):
        raise RuntimeError("service unavailable\ntraceback details")

    monkeypatch.setattr(service_execution, "execute_service_ingest_request", fail)
    outcome = run_prepared_benchmark(_prepared(tmp_path), output_dir=str(tmp_path / "artifacts"))

    assert outcome.exit_code == EXIT_INGEST_FAILURE
    assert outcome.results["failure"]["message"] == "RuntimeError: service unavailable"


def test_service_query_request_rejects_agentic_retrieval() -> None:
    with pytest.raises(HarnessRunError) as exc_info:
        build_service_query_request(
            {"query": {"top_k": 10, "agentic": True}},
            "",
            service_url="http://localhost:17670",
            service_api_token=None,
        )

    assert exc_info.value.failure.failure_reason == "invalid_benchmark_config"
    assert "service-mode" in exc_info.value.failure.message


def test_service_logs_are_registered_as_relative_artifact(tmp_path: Path) -> None:
    writer = ArtifactWriter(artifact_dir=tmp_path, run_id="run", benchmark="jp20_beir")
    writer.path("service_logs").mkdir()
    assert artifact_paths(writer)["service_logs"] == "service_logs"


def test_helm_config_requires_and_injects_immutable_image(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "helm.yaml"
    config_path.write_text("helm_chart: ./chart\n", encoding="utf-8")
    (tmp_path / "chart").mkdir()
    monkeypatch.setenv("HARNESS_HELM_SERVICE_IMAGE_REPOSITORY", "registry.example/nrl-service")
    monkeypatch.setenv("HARNESS_HELM_SERVICE_IMAGE_TAG", "main-abc123")

    config = load_helm_config(config_path)
    assert config.helm_chart == str((tmp_path / "chart").resolve())
    assert config.effective_helm_set()["service.image.repository"] == "registry.example/nrl-service"
    assert config.effective_helm_set()["service.image.tag"] == "main-abc123"


def test_helm_config_rejects_moving_image_tag(tmp_path: Path) -> None:
    config_path = tmp_path / "helm.yaml"
    config_path.write_text(
        "helm_chart: chart\nservice_image_repository: registry.example/service\nservice_image_tag: latest\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="immutable image"):
        load_helm_config(config_path)


class _FakeManager:
    def __init__(self, *, start_rc=0, stop_rc=0):
        self.start_rc = start_rc
        self.stop_rc = stop_rc
        self.calls = []

    def start(self):
        self.calls.append("start")
        return self.start_rc

    def dump_logs(self, output_dir):
        self.calls.append(("logs", Path(output_dir)))
        return 0

    def stop(self, *, uninstall=True):
        self.calls.append(("stop", uninstall))
        return self.stop_rc


def test_helm_runner_reuses_canonical_runfile_forwards_endpoint_and_always_tears_down(tmp_path: Path) -> None:
    config_path = tmp_path / "helm.yaml"
    _write_helm_config(config_path)
    manager = _FakeManager()
    commands = []
    runfile = Path(__file__).resolve().parents[1] / "harness" / "runfiles" / "jp20_beir.json"

    rc = run_helm_session(
        config_path,
        [runfile],
        output_dir=tmp_path / "session",
        manager_factory=lambda _config: manager,
        command_runner=lambda command, **_kwargs: commands.append(command) or subprocess.CompletedProcess(command, 0),
    )

    assert rc == 0
    assert manager.calls == ["start", ("stop", True)]
    endpoint_index = commands[0].index("--service-endpoint")
    assert commands[0][endpoint_index + 1] == "http://localhost:17670"
    assert "run-files" in commands[0]
    mode_index = commands[0].index("--mode")
    assert commands[0][mode_index + 1] == "service"
    assert commands[0][-1] == str(runfile)


@pytest.mark.parametrize(
    ("start_rc", "run_rc", "stop_rc", "expected_rc", "expected_prefix"),
    (
        (1, 0, 0, EXIT_HELM_FAILURE, ["start", "logs"]),
        (0, EXIT_INGEST_FAILURE, 0, EXIT_INGEST_FAILURE, ["start", "logs"]),
        (0, 0, 1, EXIT_HELM_FAILURE, ["start", "stop", "logs"]),
    ),
)
def test_helm_runner_collects_failure_logs_and_cleans_up(
    tmp_path: Path, start_rc, run_rc, stop_rc, expected_rc, expected_prefix
) -> None:
    config_path = tmp_path / "helm.yaml"
    _write_helm_config(config_path)
    manager = _FakeManager(start_rc=start_rc, stop_rc=stop_rc)

    rc = run_helm_session(
        config_path,
        [tmp_path / "run.yaml"],
        output_dir=tmp_path / "session",
        manager_factory=lambda _config: manager,
        command_runner=lambda command, **_kwargs: subprocess.CompletedProcess(command, run_rc),
    )

    names = [call[0] if isinstance(call, tuple) else call for call in manager.calls]
    assert rc == expected_rc
    assert names[: len(expected_prefix)] == expected_prefix
    assert names.count("stop") == 1


def test_port_forward_permission_error_retains_process_for_retry(monkeypatch) -> None:
    manager = object.__new__(HelmServiceManager)
    proc = SimpleNamespace(pid=123)
    manager.port_forward_processes = [proc]
    output = io.BytesIO()
    manager._port_forward_logs = {123: output}
    monkeypatch.setattr("nemo_retriever.harness.helm_manager.os.getpgid", lambda _pid: 456)
    monkeypatch.setattr(
        "nemo_retriever.harness.helm_manager.os.killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("denied")),
    )

    manager.stop_port_forwards()
    assert manager.port_forward_processes == [proc]
    assert not output.closed


def test_port_forward_waits_after_sigkill(monkeypatch) -> None:
    manager = object.__new__(HelmServiceManager)
    waits = []

    class TimedOutProcess:
        pid = 123

        def wait(_self, *, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("kubectl", timeout)

    manager.port_forward_processes = [TimedOutProcess()]
    output = io.BytesIO()
    manager._port_forward_logs = {123: output}
    signals = []
    monkeypatch.setattr("nemo_retriever.harness.helm_manager.os.getpgid", lambda _pid: 456)
    monkeypatch.setattr(
        "nemo_retriever.harness.helm_manager.os.killpg",
        lambda _pgid, sent_signal: signals.append(sent_signal),
    )

    manager.stop_port_forwards()

    assert waits == [5, 5]
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert manager.port_forward_processes == []
    assert output.closed


def test_port_forward_uses_seekable_output_and_preserves_startup_error(monkeypatch) -> None:
    manager = object.__new__(HelmServiceManager)
    manager.kubectl_cmd = ["kubectl"]
    manager.namespace = "benchmark"
    manager.local_port = 17670
    manager.remote_port = 7670
    manager.port_forward_processes = []
    manager._port_forward_logs = {}
    captured = {}

    class FailedProcess:
        pid = 123

        @staticmethod
        def poll():
            return 1

    def fake_popen(command, **kwargs):
        captured.update(command=command, seekable=kwargs["stdout"].seekable(), **kwargs)
        kwargs["stdout"].write(b"unable to listen on port 17670\n")
        return FailedProcess()

    monkeypatch.setattr("nemo_retriever.harness.helm_manager.subprocess.Popen", fake_popen)
    monkeypatch.setattr("nemo_retriever.harness.helm_manager.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="unable to listen on port 17670"):
        manager.start_port_forward("retriever")

    assert captured["stdout"] is not subprocess.PIPE
    assert captured["seekable"]
    assert captured["stderr"] is subprocess.STDOUT
    assert manager.port_forward_processes == []
    assert manager._port_forward_logs == {}
    assert captured["stdout"].closed
