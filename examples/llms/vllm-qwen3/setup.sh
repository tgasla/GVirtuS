#!/bin/bash
apt update && apt install -y python3 python3-pip
pip3 install torch==2.6.0 transformers==4.54.1 vllm==0.8.0
pip3 install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/