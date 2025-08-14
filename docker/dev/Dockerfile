FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    libxmu-dev \
    libxi-dev \
    libgl-dev \
    libosmesa6-dev \
    liblog4cplus-dev \
    librdmacm-dev \
    libibverbs-dev \
    libgtest-dev \
    liblz4-dev \
    cmake \
    nvidia-open \
    && rm -rf /var/lib/apt/lists/*

ENV GVIRTUS_HOME=/usr/local/gvirtus
ENV GVIRTUS_LOGLEVEL=0
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}

COPY ./docker/dev/build_and_run.sh /build_and_run.sh
RUN chmod +x /build_and_run.sh
ENTRYPOINT ["/build_and_run.sh"]