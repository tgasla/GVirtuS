#!/bin/bash
# phi makes use of old tranformers library. get_max_length() and seen_tokens() are not available in the latest version.
apt update && apt install -y python3 python3-pip
pip3 install transformers==4.48.2 accelerate==1.9.0 flash-attn==2.8.2