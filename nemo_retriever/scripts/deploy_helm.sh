#!/usr/bin/env bash
# Deploy (or upgrade) nemo-retriever via Helm.
#
# Usage:
#   ./nemo_retriever/scripts/deploy_helm.sh <image.repository> <image.tag> [namespace]
#
# Examples:
#   ./nemo_retriever/scripts/deploy_helm.sh nvcr.io/my-org/nemo-retriever 0.2.0
#   ./nemo_retriever/scripts/deploy_helm.sh nvcr.io/my-org/nemo-retriever 0.2.0 production
#
# Extra Helm values can be passed via HELM_EXTRA_ARGS:
#   HELM_EXTRA_ARGS="--set imagePullSecrets[0].name=ngc-secret" \
#     ./nemo_retriever/scripts/deploy_helm.sh nvcr.io/my-org/nemo-retriever 0.2.0

set -euo pipefail

REPOSITORY="${1:?Usage: $0 <image.repository> <image.tag> [namespace]}"
TAG="${2:?Usage: $0 <image.repository> <image.tag> [namespace]}"
NAMESPACE="${3:-retriever}"
RELEASE_NAME="nemo-retriever"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="${SCRIPT_DIR}/../helm"

echo "==> Deploying ${RELEASE_NAME} to namespace '${NAMESPACE}'"
echo "    image: ${REPOSITORY}:${TAG}"
echo "    chart: ${CHART_DIR}"

helm upgrade --install "${RELEASE_NAME}" "${CHART_DIR}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --set "image.repository=${REPOSITORY}" \
  --set "image.tag=${TAG}" \
  ${HELM_EXTRA_ARGS:-} \
  --wait

echo "==> Deployment complete"
echo ""
echo "To access the service:"
echo "  kubectl port-forward -n ${NAMESPACE} svc/${RELEASE_NAME} 7670:7670"
echo "  curl http://localhost:7670/health"
