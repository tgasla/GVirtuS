#!/bin/bash
apt update && apt install -y python3 python3-pip
pip install transformers==4.48.2 accelerate==1.9.0