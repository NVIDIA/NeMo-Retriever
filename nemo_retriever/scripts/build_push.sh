#!/usr/bin/env bash
# Build, tag, and push the nemo-retriever Docker image.
#
# Usage:
#   ./nemo_retriever/scripts/build_push.sh <registry> [tag]
#
# Examples:
#   ./nemo_retriever/scripts/build_push.sh nvcr.io/my-org/nemo-retriever
#   ./nemo_retriever/scripts/build_push.sh nvcr.io/my-org/nemo-retriever 0.2.0
#   ./nemo_retriever/scripts/build_push.sh 10.86.5.28:32000
#   ./nemo_retriever/scripts/build_push.sh 10.86.5.28:32000 0.2.0
#
# The build context is the repo root (one level above nemo_retriever/).

set -euo pipefail

REGISTRY="${1:?Usage: $0 <registry> [tag]}"
TAG="${2:-latest}"

# If the registry is just a host (e.g. 10.86.5.28:32000) with no image path,
# append the default image name so Docker gets a valid reference.
if [[ "${REGISTRY}" != */* ]]; then
  REGISTRY="${REGISTRY}/nemo-retriever"
fi

IMAGE="${REGISTRY}:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==> Building image from ${REPO_ROOT}"
docker build \
  -f "${REPO_ROOT}/nemo_retriever/Dockerfile" \
  -t "nemo-retriever:${TAG}" \
  "${REPO_ROOT}"

echo "==> Tagging as ${IMAGE}"
docker tag "nemo-retriever:${TAG}" "${IMAGE}"

echo "==> Pushing ${IMAGE}"
docker push "${IMAGE}"

echo "==> Done: ${IMAGE}"
