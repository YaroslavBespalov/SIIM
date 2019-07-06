#!/usr/bin/env bash

PYTHONPATH=./ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py\
    --config=./configs/config.yml\
    --paths=./configs/path.yml\
    --fold=0
