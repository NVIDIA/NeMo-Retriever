#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HELM_CONFIG="${NEMO_RETRIEVER_HELM_CONFIG:-${SCRIPT_DIR}/examples/managed-helm-main.yaml}"
RUNFILE="${NEMO_RETRIEVER_HELM_RUNFILE:-${SCRIPT_DIR}/runfiles/jp20_helm_nightly.yaml}"
OUTPUT_DIR="${NEMO_RETRIEVER_HELM_OUTPUT_DIR:-${REPO_ROOT}/artifacts/helm-jp20-$(date -u +%Y%m%d_%H%M%S_UTC)}"

: "${HARNESS_HELM_SERVICE_IMAGE_REPOSITORY:?set the main/nightly service image repository}"
: "${HARNESS_HELM_SERVICE_IMAGE_TAG:?set the immutable main/nightly service image tag}"

exec uv run --project "${REPO_ROOT}/nemo_retriever" \
  python -m nemo_retriever.harness.helm_runner \
  --config "${HELM_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --session-name helm_jp20 \
  "${RUNFILE}" "$@"
