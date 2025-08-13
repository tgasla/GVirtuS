#!/bin/bash
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib
export GVIRTUS_LOGLEVEL=50000
LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcuda.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcublas.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcublasLt.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcudnn.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcufft.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcurand.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcusparse.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcusolver.so: \
    ${GVIRTUS_HOME}/lib/frontend/libnvrtc.so" \
python3 phi.py