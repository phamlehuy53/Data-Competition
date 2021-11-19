#!/bin/bash

# train then evaluate multiple times
# log the evalutaion metrics for statistics: csv: start_time | version | train/val | mp | mr | wap50 | map50 | map | loss 

date

num_train=$1
echo "Train model for $num_train times"
python3 train.py --batch-size 64 --device 0 --name $num_train
