#!/bin/bash

source .env/bin/activate

# nccl log
export NCCL_DEBUG=INFO
# nvlink
nvidia-smi nvlink --status

# log dir
mkdir -p outputs/logs/1.3B

now=$(date +"%Y-%m-%d-%H-%M-%S")

python ./deepy.py train.py \
  --hostfile scripts/1.3B/hostfile \
  -d configs 1-3B.yml multi-node/1.3B/dp16-tp1-pp1.yml \
  &> outputs/logs/1.3B/2node-dp16-tp1-pp1-$now.log
