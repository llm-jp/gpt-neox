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
  -d configs 1node/2.7b-dp4-tp1-pp2.yml \
  &> outputs/logs/2.7B/1node-dp4-tp1-pp2-$now.log
