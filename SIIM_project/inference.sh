#!/usr/bin/env bash

PYTHONPATH=./ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 python ./src/inference.py\
    --config=./configs/config_inference.yml\
    --paths=./configs/path_for_test.yml
