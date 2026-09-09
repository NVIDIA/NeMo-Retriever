# Run the BO767 4-GPU, 2x NIM Helm benchmark

This example runs the BO767 text-only service qualification with Helm chart
`26.08.2`, NRL service image `26.08.1`, a hybrid LanceDB index, automatic
retrieval, and element-level text embeddings. It creates two replicas of each
core NIMService: page-elements, table-structure, OCR, and VL embed. The
deployment therefore requests eight GPU slots.

Use this configuration on a node with four physical GPUs only after confirming
that two colocated NIM workloads fit in each GPU's memory. NVIDIA GPU Operator
time slicing advertises logical slots. It does not isolate GPU memory or
guarantee that exactly two pods land on each physical GPU.

## Prerequisites

Install the NVIDIA GPU Operator, Helm, kubectl, uv, and the NRL checkout. The
cluster administrator must enable the GPU sharing configuration. Set the NGC
API key only in your shell. Do not add it to any repository file.

```bash
export RETRIEVER_CHECKOUT=/local/path/to/NeMo-Retriever
export NGC_API_KEY=<your-NGC-API-key>
export HELM_NAMESPACE=nemo-retriever-bo767-vl-text-2608-4gpu-2x
export GPU_OPERATOR_NAMESPACE=gpu-operator
cd "$RETRIEVER_CHECKOUT"
uv sync --project nemo_retriever
```

## Configure two GPU slots per physical GPU

Apply the supplied ConfigMap and select it in the GPU Operator ClusterPolicy.
The `default: "any"` setting affects every eligible GPU Operator node. On a
mixed cluster, omit `default` and label only the intended benchmark node.

```bash
kubectl apply -n "$GPU_OPERATOR_NAMESPACE" \
  -f nemo_retriever/harness/examples/gpu-operator-time-slicing-2x.yaml

kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  -n "$GPU_OPERATOR_NAMESPACE" --type merge \
  -p '{"spec":{"devicePlugin":{"config":{"name":"nemo-retriever-time-slicing-2x","default":"any"}}}}'

kubectl get nodes \
  -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
```

A four-GPU target node must report eight allocatable `nvidia.com/gpu` slots.

## Create the namespace and NGC Secrets

```bash
helm repo add nvstaging https://helm.ngc.nvidia.com/nvstaging/nim \
  --username '$oauthtoken' --password "$NGC_API_KEY"
helm repo update

kubectl create namespace "$HELM_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$HELM_NAMESPACE" create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$HELM_NAMESPACE" create secret generic ngc-api \
  --from-literal=NGC_API_KEY="$NGC_API_KEY" \
  --from-literal=NGC_CLI_API_KEY="$NGC_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -
```

For MicroK8s, use `microk8s helm` and `microk8s kubectl`, and set
`helm_bin`, `kubectl_bin`, `helm_sudo`, and `kubectl_sudo` in a machine-local
copy of the managed Helm configuration.

## Run the benchmark

Create a machine-local dataset map. Set `datasets.bo767.path` to the BO767
corpus and `datasets.bo767.query_file` to its matching query CSV file.

```bash
export DATASET_PATHS=/local/path/to/dataset-paths.yaml
cp nemo_retriever/harness/dataset_paths.example.yaml "$DATASET_PATHS"

export RUN_DIR=/local/path/to/retriever-artifacts/bo767-4gpu-2x-$(date -u +%Y%m%d_%H%M%S_UTC)

uv run --project nemo_retriever retriever-harness run-helm \
  --config nemo_retriever/harness/examples/managed-helm-bo767-vl-text-26.08.2-4gpu-2x.yaml \
  --session-name bo767_vl_text_helm_4gpu_2x \
  --output-dir "$RUN_DIR" \
  --dataset-paths "$DATASET_PATHS" \
  nemo_retriever/harness/runfiles/bo767_vl_text_hybrid_beir_service.json
```

## Verify the deployment and result

Before ingest starts, confirm that eight core NIM pods are Ready and each
requests one GPU slot.

```bash
kubectl -n "$HELM_NAMESPACE" get pods
kubectl -n "$HELM_NAMESPACE" get nimservice
```

After completion, inspect `session_summary.json` and retain every child
`results.json`, `environment.json`, `resolved_benchmark.json`, `ingest_plan.json`,
`query_plan.json`, `beir_metrics.json`, and `run.log`. A valid BO767 run has
`files == 767`, `pages == 54730`, and `query_count == 991`.

Compare `pages_per_sec_ingest`, `recall_10`, and `ndcg_10` between runs with
identical image tags, chart version, dataset, and runfile. Higher GPU slot
count does not by itself guarantee higher ingest throughput. Use the retained
plans and logs to identify the limiting NIMService.
