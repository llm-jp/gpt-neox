#!/bin/bash

source .env/bin/activate

# nccl log
export NCCL_DEBUG=INFO
# nvlink
nvidia-smi nvlink --status


python ./deepy.py train.py \
  --hostfile hostfile \
  -d configs 2-7B.yml mdx-config/multi-node/gpt-2.7b-dp16.yml
