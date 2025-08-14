## This Dockerfile is for building a Gvirtus image with CUDA support.
## This image has already GVirtuS installed, so it is not expected to be mounted at runtime (useful for production).
## The image will be pushed at taslanidis/gvirtus:cuda12.6.3-cudnn-ubuntu22.04 using the `make docker-build-push-prod`

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
ENV GVIRTUS_LOGLEVEL=20000
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}

COPY cmake /gvirtus/cmake
COPY etc /gvirtus/etc
COPY include /gvirtus/include
COPY plugins /gvirtus/plugins
COPY src /gvirtus/src
COPY tools /gvirtus/tools
COPY tests /gvirtus/tests
COPY CMakeLists.txt /gvirtus/CMakeLists.txt

RUN mkdir gvirtus/build && cd gvirtus/build && cmake .. && make && make install
ENTRYPOINT ["/bin/bash", "-c", "${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties.json"]