#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates git git-lfs build-essential ninja-build patchelf \
  software-properties-common python3-pip
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends \
  python3.11 python3.11-dev python3.11-venv \
  python3.13 python3.13-dev python3.13-venv
git lfs install
arch_env_arg=""
if [[ "$(uname -m)" == "aarch64" ]]; then
  arch_env_arg="--build-env ARCH=arm64"
fi
mkdir -p dist-out/nemotron-ocr-v2
for PY_VER in 3.11 3.13; do
  PYTHON="python${PY_VER}"
  TAG="cp${PY_VER//./}"
  STAGING="dist-staging-${TAG}"
  VENV_DIR="/tmp/.venv-build-${TAG}"
  echo "=== Building nemotron-ocr for Python ${PY_VER} ==="
  "${PYTHON}" --version
  "${PYTHON}" -m pip install --break-system-packages "packaging>=24"
  "${PYTHON}" ci/scripts/nightly_build_publish.py \
    --repo-id nemotron-ocr-v2 \
    --repo-url https://huggingface.co/nvidia/nemotron-ocr-v2 \
    --work-dir ".work-${TAG}" \
    --dist-dir "${STAGING}" \
    --venv-dir "${VENV_DIR}" \
    --project-subdir nemotron-ocr \
    --nightly-base-version 2.0.1 \
    --set-requires-python ">=3.11,<3.14" \
    --skip-sdist \
    --hatch-force-platform-wheel \
    --auditwheel-repair \
    --auditwheel-exclude libtorch_cpu.so \
    --auditwheel-exclude libtorch_cuda.so \
    --auditwheel-exclude libtorch.so \
    --auditwheel-exclude libc10.so \
    --auditwheel-exclude libc10_cuda.so \
    --auditwheel-exclude libtorch_python.so \
    --build-no-isolation \
    --venv-pip-install hatchling \
    --venv-pip-install "setuptools>=68" \
    --venv-pip-install ninja \
    --venv-pip-install "torch==${OCR_TORCH_VERSION}" \
    --venv-pip-install "torchvision==${OCR_TORCHVISION_VERSION}" \
    --pin-runtime-dependency torch \
    --pin-runtime-dependency torchvision \
    --build-env BUILD_CPP_EXTENSION=1 \
    --build-env BUILD_CPP_FORCE=1 \
    ${arch_env_arg}
  cp "${STAGING}/nemotron-ocr-v2/"*.whl dist-out/nemotron-ocr-v2/
done
ls -lh dist-out/nemotron-ocr-v2/
