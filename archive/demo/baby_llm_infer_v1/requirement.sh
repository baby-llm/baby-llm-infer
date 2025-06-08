#!/bin/bash

source /etc/network_turbo
pip install torch transformers bitsandbytes accelerate
pip install flash-attn --no-build-isolation
pip install huggingface_hub
huggingface-cli login