## This Dockerfile is for building a GVirtuS image with CUDA support.
## It does not include the Gvirtus source code, as it is expected to be mounted at runtime (useful for development).
## The image will be pushed at taslanidis/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04 using the `make docker-build-push-dev` command.

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    build-essential \
    libxmu-dev \
    libxi-dev \
    libgl-dev \
    libosmesa-dev \
    liblog4cplus-dev \
    librdmacm-dev \
    libibverbs-dev \
    libgtest-dev \
    liblz4-dev \
    cmake \
    nvidia-driver-550 \
    && rm -rf /var/lib/apt/lists/*

ENV GVIRTUS_HOME=/usr/local/gvirtus
ENV GVIRTUS_LOGLEVEL=0
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}

COPY ./docker/dev/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]