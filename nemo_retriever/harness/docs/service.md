<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Service Harness Runs

Service execution runs the same registered benchmarks through Retriever service
ingest and query APIs. It writes the same artifacts and evaluates the same
metric gates as library execution.

Choose how the service is provided:

| Target | Command |
| --- | --- |
| An already-running service | `retriever harness run-files --mode service` |
| A temporary service provisioned by the harness | `retriever harness run-helm` |

Helm is a provisioning wrapper, not a fourth benchmark mode.

## Test an Existing Service

The service must be reachable from the harness host. Export its bearer token
only when authentication is enabled:

```bash
export HARNESS_SERVICE_API_TOKEN=...
```

Run a checked-in benchmark against the endpoint:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --mode service \
  --service-endpoint http://localhost:7670 \
  --session-name service_jp20 \
  --output-dir /local/path/to/retriever-artifacts/service-jp20 \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

`--mode service` overrides the runfile mode for the session.
`--service-endpoint` applies only to service-mode children, so mixed sessions
remain possible. If omitted, the endpoint defaults to
`http://localhost:7670`.

The dataset paths identify documents and queries available to the harness
client. Service storage and infrastructure remain owned by the service.

## Provision a Service with Helm

`run-helm` deploys one immutable service image, waits for readiness, establishes
a local port-forward, runs the shared `run-files` session in service mode,
collects logs on failure, and tears the release down.

Prerequisites:

- working `helm` and `kubectl` commands, or equivalents selected in the config
- a Kubernetes cluster with the required image-pull and NGC secrets
- an immutable service image built from the checkout
- a non-secret Helm deployment file

Start from
[`examples/managed-helm-main.yaml`](../examples/managed-helm-main.yaml). Select a
checked-in profile such as [`helm-profiles/core.yaml`](../helm-profiles/core.yaml)
with `helm_values_file`.

Set the immutable image:

```bash
export HARNESS_HELM_SERVICE_IMAGE_REPOSITORY=nvcr.io/example/nrl-service
export HARNESS_HELM_SERVICE_IMAGE_TAG=<immutable-tag>
```

Then run:

```bash
uv run --project nemo_retriever retriever harness run-helm \
  --config nemo_retriever/harness/examples/managed-helm-main.yaml \
  --output-dir /local/path/to/retriever-artifacts/helm-jp20 \
  --session-name helm_jp20 \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

The deployment file may configure the chart, release, namespace, values file,
`helm_set` overrides, timeouts, local service port, command paths, and whether
Helm or kubectl requires `sudo`. Keep credentials out of it.

Image tags named `latest`, `main`, or `nightly` are rejected because they are
not reproducible. `run-helm` returns the benchmark session status unless
deployment or teardown fails, in which case it returns `4`.

## Inspect and Report Results

Read `session_summary.json` first. Failed managed sessions may also contain
`service_logs/`. The artifact tree records the benchmark configuration and
runtime environment. Keep the immutable image reference and non-secret Helm
configuration with the surrounding job record.

Reporting is a separate step:

```bash
uv run --project nemo_retriever retriever harness post-slack \
  --preview \
  /local/path/to/retriever-artifacts/helm-jp20
```

See the [shared artifact and reporting contract](../README.md#results-and-artifacts)
for exit codes, Slack posting, and release comparisons.
