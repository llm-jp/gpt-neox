#!/bin/bash

source .env/bin/activate

# nccl log
export NCCL_DEBUG=INFO
# nvlink
nvidia-smi nvlink --status

# log dir
mkdir -p outputs/logs/2.7B

now=$(date +"%Y-%m-%d-%H-%M-%S")

python ./deepy.py train.py \
  -d configs 2-7B.yml 1node/2.7b-dp8-tp1-pp1.yml \
  &> outputs/logs/2.7B/1node-dp8-tp1-pp1-$now.log
