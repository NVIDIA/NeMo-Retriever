ARG ELASTICSEARCH_VERSION=9.5.2
FROM docker.elastic.co/elasticsearch/elasticsearch:${ELASTICSEARCH_VERSION}

USER root

ENV CUVS_VERSION=26.04.00.194111
ENV CUDA_VERSION=12.9.1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV RAFT_DEBUG_LOG_FILE=/dev/null
ENV LIBCUVS_DIR=/opt/cuvs
ENV LIBCUVS_ARCHIVE_DIR=libcuvs-linux-x86_64-${CUVS_VERSION}_cuda12-archive
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=${LIBCUVS_DIR}/${LIBCUVS_ARCHIVE_DIR}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

RUN microdnf install -y dnf \
    && dnf install -y 'dnf-command(config-manager)' \
    && dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf upgrade -y \
    && dnf install -y \
        cuda-cudart-12-9 \
        cuda-compat-12-9 \
        cuda-libraries-12-9 \
        cuda-nvtx-12-9 \
        libnccl-2.27.3-1+cuda12.9 \
        tar \
        xz \
    && dnf clean all \
    && rm -rf /var/cache/yum/*

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && mkdir -p "${LIBCUVS_DIR}" \
    && chmod 775 "${LIBCUVS_DIR}" \
    && cd "${LIBCUVS_DIR}" \
    && CUVS_ARCHIVE="${LIBCUVS_ARCHIVE_DIR}.tar.xz" \
    && curl -fO "https://developer.download.nvidia.com/compute/cuvs/redist/libcuvs/linux-x86_64/${CUVS_ARCHIVE}" \
    && tar -xJf "${CUVS_ARCHIVE}" \
    && rm -f "${CUVS_ARCHIVE}"

USER 1000:0
