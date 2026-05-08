# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3
#
# Build from repo root: docker build -f nemo_retriever/Dockerfile -t nemo-retriever .
# Run: docker run nemo-retriever  (shell with venv active)
# Run with dev mount: docker run -v $(pwd):/workspace -it nemo-retriever   (code changes reflect without rebuild)
# Run with data:     docker run -v /host/docs:/data nemo-retriever /data

ARG BASE_IMG=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMG_TAG=jammy-20250619

FROM $BASE_IMG:$BASE_IMG_TAG AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
      libglib2.0-0 \
      wget \
    && apt-get clean

# ffmpeg/ffprobe for audio extraction (run before LibreOffice so apt state is consistent)
COPY docker/scripts/install_ffmpeg.sh /tmp/install_ffmpeg.sh
RUN bash /tmp/install_ffmpeg.sh && rm /tmp/install_ffmpeg.sh

# LibreOffice (headless) for docx/pptx -> PDF. GPL source handling per nv-ingest Dockerfile.
ARG GPL_LIBS="\
    libltdl7 \
    libhunspell-1.7-0 \
    libhyphen0 \
    libdbus-1-3 \
"
# Keep libfreetype6 so LibreOffice (soffice.bin) can load; omit from force-remove.
ARG FORCE_REMOVE_PKGS="\
    ucf \
    liblangtag-common \
    libjbig0 \
    pinentry-curses \
    gpg-agent \
    gnupg-utils \
    gpgsm \
    gpg-wks-server \
    gpg-wks-client \
    gpgconf \
    gnupg \
    readline-common \
    libreadline8 \
    dirmngr \
    libjpeg8 \
"
RUN sed -i 's/# deb-src/deb-src/' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      dpkg-dev \
      libreoffice \
      $GPL_LIBS \
    && apt-get source $GPL_LIBS \
    && for pkg in $FORCE_REMOVE_PKGS; do \
         dpkg --remove --force-depends "$pkg" || true; \
       done \
    && apt-get clean

ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV UV_LINK_MODE=copy
# Keep uv-managed Python outside /root so the non-root service user can execute
# venv console-script interpreters.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 \
    && uv venv --python 3.12 /opt/retriever_runtime

RUN --mount=type=cache,target=/root/.cache/uv \
    . /opt/retriever_runtime/bin/activate \
    && wget -qO- https://bootstrap.pypa.io/get-pip.py | python -

RUN --mount=type=cache,target=/root/.cache/uv \
    . /opt/retriever_runtime/bin/activate \
    && pip install --no-cache-dir openai

RUN --mount=type=cache,target=/root/.cache/uv \
    . /opt/retriever_runtime/bin/activate \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt update && apt-get --fix-broken install -y && apt-get -y install cuda-toolkit-13-0

WORKDIR /workspace
COPY data data
COPY nemo_retriever nemo_retriever

# ENV VIRTUAL_ENV=/opt/retriever_runtime
# ENV PATH=/opt/retriever_runtime/bin:/root/.local/bin:$PATH
# ENV LD_LIBRARY_PATH=/opt/retriever_runtime/lib:${LD_LIBRARY_PATH}

# ---------------------------------------------------------------------------
# Install nemo_retriever and path deps (build context = repo root)
# To pick up dev changes without rebuilding, run with:
#   -v /path/to/NeMo-Retriever/main:/workspace
# The editable install points at /workspace, so the mounted tree is used.
# ---------------------------------------------------------------------------
FROM base AS install

WORKDIR /workspace

# Unbuffered stdout/stderr so CLI output appears when run without a TTY (e.g. docker run without -it)
ENV PYTHONUNBUFFERED=1

# Activate venv by default so CLI and python see nemo_retriever; mount over /workspace for dev.
ENV VIRTUAL_ENV=/opt/retriever_runtime
ENV PATH=/opt/retriever_runtime/bin:$PATH

# Editable install: at runtime, -v host_repo:/workspace overrides these dirs so dev changes apply.
SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    . /opt/retriever_runtime/bin/activate \
    && uv pip install -e ./nemo_retriever

# Default: run in-process pipeline (help if no args)
CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# Service profile: run the FastAPI ingest service.
#
# Build:  docker build -f nemo_retriever/Dockerfile --target service \
#             -t nemo-retriever-service .
#
# Run with the bundled default config:
#   docker run --rm -p 7670:7670 nemo-retriever-service
#
# Run with a custom config mounted at the well-known path:
#   docker run --rm -p 7670:7670 \
#     -v /host/path/to/retriever-service.yaml:/etc/nemo-retriever/retriever-service.yaml:ro \
#     nemo-retriever-service
#
# The container always loads its config from
#   /etc/nemo-retriever/retriever-service.yaml
# Bind-mount your file to that exact path to override the bundled default.
# ---------------------------------------------------------------------------
FROM install AS service

ENV NEMO_RETRIEVER_SERVICE_CONFIG=/etc/nemo-retriever/retriever-service.yaml

ENV PATH=/opt/retriever_runtime/bin:$PATH

RUN chmod a+rx /usr/local/bin/uv /usr/local/bin/uvx \
    && chmod -R a+rX /opt/uv \
    && groupadd -r nemo && useradd -r -g nemo -d /workspace -s /sbin/nologin nemo \
    && mkdir -p /etc/nemo-retriever /var/lib/nemo-retriever \
    && cp /workspace/nemo_retriever/src/nemo_retriever/service/retriever-service.yaml \
            "${NEMO_RETRIEVER_SERVICE_CONFIG}" \
    && chown -R nemo:nemo /workspace /etc/nemo-retriever /var/lib/nemo-retriever /opt/retriever_runtime

EXPOSE 7670

USER nemo

CMD ["retriever", "service", "start", "--config", "/etc/nemo-retriever/retriever-service.yaml"]
