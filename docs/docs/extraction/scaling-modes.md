# Resource Scaling Modes for NeMo Retriever Library

This guide covers how resource scaling modes work across stages in [NeMo Retriever Library](overview.md), and how to configure it when you deploy with the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md).

- **Static scaling**: Each pipeline stage runs a fixed number of replicas based on heuristics (memory-aware). Good for consistent latency; higher steady-state memory usage.
- **Dynamic scaling**: Only the source stage is fixed; other stages scale up/down based on observed resource pressure. Better memory efficiency; may briefly pause to spin replicas back up after idle periods.

## When to choose which

- **Choose Static** when latency consistency and warm pipelines matter more than memory minimization.
- **Choose Dynamic** when memory headroom is constrained or workloads are bursty/idle for long periods.

## Configure (Helm / Kubernetes)

Set environment variables on the **ingestion runtime** workload that exposes the NeMo Retriever Library API (the chart’s ingestion / `*-ms-runtime` deployment or equivalent). Use Helm `values.yaml`, `extraEnv`, or a `Secret`/`ConfigMap` reference as described in the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md).

### Select mode

- **Dynamic (default)**
  - `INGEST_DISABLE_DYNAMIC_SCALING=false`
  - `INGEST_DYNAMIC_MEMORY_THRESHOLD=0.80` (fraction of memory; worker scaling reacts around this level)

- **Static**
  - `INGEST_DISABLE_DYNAMIC_SCALING=true`
  - Optionally set a static memory threshold:
    - `INGEST_STATIC_MEMORY_THRESHOLD=0.85` (fraction of total memory reserved for static replicas)

Example (Static) — illustrative `env` fragment for the ingestion runtime:

```yaml
env:
  - name: INGEST_DISABLE_DYNAMIC_SCALING
    value: "true"
  - name: INGEST_STATIC_MEMORY_THRESHOLD
    value: "0.85"
```

Example (Dynamic):

```yaml
env:
  - name: INGEST_DISABLE_DYNAMIC_SCALING
    value: "false"
  - name: INGEST_DYNAMIC_MEMORY_THRESHOLD
    value: "0.80"
```

### Pipeline config mapping

- `pipeline.disable_dynamic_scaling` ⇐ `INGEST_DISABLE_DYNAMIC_SCALING`
- `pipeline.dynamic_memory_threshold` ⇐ `INGEST_DYNAMIC_MEMORY_THRESHOLD`
- `pipeline.static_memory_threshold` ⇐ `INGEST_STATIC_MEMORY_THRESHOLD`

## Trade-offs recap

- **Dynamic**
  - Pros: Better memory efficiency; stages scale down when idle; can force scale-down under spikes.
  - Cons: After long idle, stages may scale to 0 replicas causing brief warm-up latency when work resumes.

- **Static**
  - Pros: Stable, predictable latency; stages remain hot.
  - Cons: Higher baseline memory usage over time.

## Sources of memory utilization

- **Workload size and concurrency**
  - More in‑flight jobs create more objects (pages, images, tables, charts) and large artifacts (for example, embeddings).
  - Example: 1 MB text file → paragraphs with 20% overlap → 4k‑dim embeddings base64‑encoded to JSON
    - Assumptions: ~600 bytes per paragraph. 20% overlap ⇒ effective step ≈ 480 bytes. Chunks ≈ 1,000,000 / 480 ≈ 2,083.
    - Per‑embedding size: 4,096 dims × 4 bytes (float32) = 16,384 bytes; base64 expansion × 4/3 ≈ 21,845 bytes (≈21.3 KB).
    - Total embeddings payload: ≈ 2,083 × 21.3 KB ≈ 45 MB, excluding JSON keys/metadata.
    - Takeaway: a 1 MB source can yield ≳40× memory just for embeddings, before adding extracted text, images, or other artifacts.
  - Example: PDF rendering and extracted images (A4 @ 72 DPI)
    - Rendering a page is a large in‑memory buffer; each extracted sub‑image adds more, and base64 inflates size.
    - Page pixels ≈ 8.27×72 by 11.69×72 ≈ 595×842 ≈ 0.50 MP.
    - RGB (3 bytes/pixel) ≈ 1.5 MB per page buffer; RGBA (4 bytes/pixel) ≈ 2.0 MB.
    - Ten 1024×1024 RGB crops ≈ 3.0 MB each in memory → base64 (+33%) ≈ 4.0 MB each ⇒ ~40 MB just for crops (JSON not included).
    - If you also base64 the full page image, expect another ~33% over the raw byte size (compression varies by format).
- **Library behavior**
  - Components like PyArrow may retain memory longer than expected (delayed free).
- **Queues and payloads**
  - Base64‑encoded, fragmented documents in Redis consume memory proportional to concurrent jobs, clients, and drain speed.

## Where to look in Helm values

Open your release’s `values.yaml` (or the chart’s documented path for the ingestion runtime) and set:

- `INGEST_DISABLE_DYNAMIC_SCALING`
- `INGEST_DYNAMIC_MEMORY_THRESHOLD`
- `INGEST_STATIC_MEMORY_THRESHOLD`

Exact keys depend on chart version; align with the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md).



## Related Topics

- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Troubleshooting](troubleshoot.md)
