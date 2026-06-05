# Tracing and Zipkin Parity for Current Helm

## Context

The 26.1 non-NRL Helm chart exposed OpenTelemetry traces through Zipkin in a way
that downstream performance tooling could query after an ingest run. The current
`nemo_retriever/helm` chart still deploys an OpenTelemetry collector, but its
default collector configuration exports traces only to the debug exporter, the
chart no longer deploys Zipkin, service pods do not receive chart-managed OTel
environment variables, and chart-managed NIMServices do not inherit NIM/Triton
OTel configuration.

The regression surfaced in the Slack thread about BCS perf analysis: the current
26.05 Helm chart does not provide the trace results that earlier `nv-ingest`
Helm releases exposed. This spec is based on fresh `upstream/main` commit
`56daf782` as of 2026-06-05 and targets parity with, or improvement over, the
26.1 trace workflow.

## Goals

- Restore the 26.1-style performance workflow: after a Helm-deployed ingest run,
  a user can collect a trace id and query Zipkin at `/api/v2/trace/{trace_id}`.
- Expose spans that are at least as useful as 26.1 for understanding what
  happened behind an ingest request and how long each step took.
- Support current service architecture, including standalone and split
  gateway/realtime/batch topologies.
- Apply tracing defaults generically across chart-managed NIMServices that can
  emit OTel, not only the NIMs used in the default PDF path.
- Degrade safely when tracing infrastructure is disabled or unreachable.

## Non-Goals

- Replacing Prometheus metrics or the current HPA/autoscaling metric work.
- Redesigning a multi-backend observability platform beyond Zipkin parity.
- Changing BCS scripts directly.
- Guaranteeing that every NIM image supports every inherited OTel environment
  variable; the chart must provide opt-out and override paths for incompatible
  NIMs.

## Recommended Approach

Implement end-to-end tracing parity, not chart plumbing only.

Adding Zipkin and collector export configuration alone would restore visible
Helm objects, but it would not guarantee meaningful spans because the current
service does not initialize and propagate OpenTelemetry traces the way the old
`nv-ingest` service did. The implementation should therefore cover Helm
resources, service trace initialization, context propagation, and spans around
the current ingest pipeline.

## Helm Design

### Zipkin Topology

Add a chart-owned `topology.zipkin` values block. When `topology.otel.enabled` is
true, Zipkin should be enabled by default unless the operator explicitly disables
it. The chart should render:

- A Zipkin Deployment.
- A Zipkin ClusterIP Service exposing port `9411`.
- Resource, image, node selector, toleration, and affinity values consistent
  with the chart's existing `topology.otel` style.

Operators should be able to disable the in-cluster Zipkin deployment while still
keeping OTel enabled, for example when exporting to an external Zipkin or OTLP
backend.

### Collector Export

Update the default `topology.otel.config` so the traces pipeline exports to both
`debug` and `zipkin`. The Zipkin exporter endpoint should be rendered from Helm
helpers, not hard-coded to a release name:

```yaml
exporters:
  debug:
    verbosity: basic
  zipkin:
    endpoint: "http://RENDERED_FULLNAME-zipkin:9411/api/v2/spans"
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug, zipkin]
```

The implementation may add the old 26.1 transform and tail-sampling processors
if needed to filter health checks and keep trace names readable, but that should
not obscure the primary parity goal.

### Service OTel Environment

Add a Helm helper that renders service tracing environment variables and include
it in both service Deployment env blocks:

- Standalone mode container env.
- Split mode gateway/realtime/batch container env.

Default values should include:

- `OTEL_EXPORTER_OTLP_ENDPOINT=http://RENDERED_FULLNAME-otel:4317`
- `OTEL_SERVICE_NAME=nemo-retriever-service`
- `OTEL_TRACES_EXPORTER=otlp`
- `OTEL_METRICS_EXPORTER=otlp`
- `OTEL_LOGS_EXPORTER=none`
- `OTEL_PROPAGATORS=tracecontext,baggage`
- `OTEL_PYTHON_EXCLUDED_URLS=health`

The helper should avoid duplicate env names when users provide explicit
overrides via `service.env`. User-provided values should win.

### Generic NIM OTel Environment

Add chart-wide NIM tracing defaults, for example under `nimOperator.otel`, and
render them into every chart-managed NIMService unless that NIM opts out or
overrides them.

Suggested defaults:

- `NIM_ENABLE_OTEL=true`
- `NIM_OTEL_SERVICE_NAME=RENDERED_NIM_SERVICE_NAME`
- `NIM_OTEL_TRACES_EXPORTER=otlp`
- `NIM_OTEL_METRICS_EXPORTER=console`
- `NIM_OTEL_EXPORTER_OTLP_ENDPOINT=http://RENDERED_FULLNAME-otel:4318`
- `TRITON_OTEL_URL=http://RENDERED_FULLNAME-otel:4318/v1/traces`
- `TRITON_OTEL_RATE=1`

Per-NIM service names should default to the rendered NIMService name so Zipkin
traces are readable. Each NIM block should be able to disable inherited tracing
or override individual env vars without affecting the rest of the chart.

## Service Runtime Design

### OpenTelemetry Initialization

The service should explicitly initialize OpenTelemetry at startup when tracing is
enabled by environment:

- `OTEL_TRACES_EXPORTER=otlp`
- `OTEL_EXPORTER_OTLP_ENDPOINT` is present and non-empty

Initialization should create a tracer provider with service resource attributes
that identify the service role (`standalone`, `gateway`, `realtime`, or
`batch`). Export should use OTLP gRPC to the in-cluster collector. If exporter
setup fails, the service should continue and log the tracing failure.

OTel packages should be direct dependencies of the service if code imports them
directly, rather than relying on transitive lockfile entries.

### Trace ID Contract

Create a stable root trace per ingest job. `POST /v1/ingest/job` should expose
the trace id in both a `trace_id` response field and an `x-trace-id` response
header. Subsequent document/page/whole submissions for the job should attach to
that root trace.

The trace id must be suitable for the historical Zipkin lookup pattern:

```text
/api/v2/trace/{trace_id}
```

When the request is routed through split topology, gateway-to-worker requests
should carry W3C trace context (`traceparent`) plus the existing job/document
routing identifiers so worker spans attach to the same trace.

### Span Coverage

The first implementation should emit spans around:

- Job creation.
- Document/page/whole upload handling.
- Gateway routing decision and proxy forwarding.
- Queue admission and enqueue rejection.
- Queue wait.
- Realtime/batch pool processing.
- Pipeline execution.
- Key pipeline stages, using existing stage/operator names where available.
- Remote NIM HTTP/gRPC calls.

Existing internal `execution_trace_log` data can be bridged into child spans
where practical. The first-class contract, however, should be OpenTelemetry
spans exported through the collector into Zipkin.

### Span Attributes

Spans should include useful, non-secret attributes:

- `service.role`
- `pool`
- `job.id`
- `document.id`
- `route`
- `stage`
- `nim.service`
- `http.status_code` or gRPC status when available
- `error.type` and sanitized error classification when available

Trace context and attributes must not include API keys, bearer tokens, NGC
tokens, request bodies, or other credentials.

## Error Handling and Compatibility

Tracing must not become part of the ingest success path. If Zipkin or the OTel
collector is disabled, unavailable, slow, or misconfigured, ingest should
continue. Export failures should be logged at warning or debug level.

NIM tracing should also be safe by default. If a NIM image rejects or ignores
inherited OTel env vars, the operator can disable inherited NIM tracing for that
model or override the problematic env vars. This should not disable tracing for
other NIMs.

The exact span names do not need to match 26.1, because the current service
architecture differs. The compatibility contract is that users can query Zipkin
by trace id and understand the timing of service, routing, queue, pipeline, and
remote NIM work at least as well as they could in 26.1.

## Testing

### Helm Render Tests

Add tests that render the Helm chart and assert:

- Zipkin resources exist when OTel is enabled by default.
- Zipkin resources are absent when `topology.zipkin.enabled=false`.
- The collector traces pipeline includes the Zipkin exporter.
- The Zipkin exporter endpoint uses the rendered chart fullname.
- Standalone and split service Deployments receive OTel env vars.
- User-provided `service.env` values override chart defaults without duplicate
  env names.
- Every enabled chart-managed NIMService inherits the NIM/Triton OTel env block.
- Per-NIM tracing opt-out and override values render correctly.

### Python Unit Tests

Add tests that validate:

- Service startup configures an OTLP tracer provider only when tracing is
  enabled.
- Job creation returns or exposes a Zipkin-compatible trace id.
- Document/page/whole submissions attach to an existing job trace.
- Gateway-to-worker proxy calls propagate W3C trace context.
- Queue and pool processing spans include expected attributes.
- Remote NIM client spans include stage/service/status attributes and do not
  expose credentials.

### Smoke Test

Add a documented manual or CI-optional smoke test that:

1. Installs a minimal Helm profile with OTel and Zipkin enabled.
2. Submits a small ingest job.
3. Collects the returned trace id.
4. Queries Zipkin `/api/v2/trace/{trace_id}`.
5. Verifies the trace includes a service root span, routing/queue/pool spans,
   pipeline-stage spans, and any NIM/Triton spans available from the enabled
   NIM images.

This can remain CI-optional if cluster and NIM dependencies make it too
expensive for standard CI.

## Documentation

Update Helm documentation to explain:

- How to enable or disable chart-managed Zipkin.
- How to export to an external Zipkin or OTLP backend.
- How to port-forward Zipkin.
- How to collect a trace id from an ingest job.
- How to query `/api/v2/trace/{trace_id}`.
- Known limitations for NIM images that do not support inherited OTel env vars.

## Acceptance Criteria

- A Helm user can reproduce the 26.1-style performance workflow on the current
  chart.
- Zipkin is available from a default OTel-enabled Helm install unless explicitly
  disabled.
- A trace id returned or exposed by an ingest job can be queried from Zipkin at
  `/api/v2/trace/{trace_id}`.
- The trace contains meaningful spans for the service request, gateway/worker
  handoff where applicable, queue/pool processing, pipeline work, and remote NIM
  calls.
- Chart-managed NIMServices inherit tracing env vars generically, with per-NIM
  opt-out and override support.
- Tracing failures do not fail ingest requests.
- Tests cover Helm rendering, trace propagation, and span attributes.
