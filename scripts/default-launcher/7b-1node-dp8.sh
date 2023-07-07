#!/bin/bash

source .env/bin/activate

python ./deepy.py train.py -d configs 2-7B.yml mdx-config/1node/gpt2-7b-dp8.yml
