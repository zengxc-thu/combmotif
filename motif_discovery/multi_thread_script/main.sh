#!/bin/bash

export CUDA_VISIBLE_DEVICES=0  # which gpu to use
export PYTHONPATH=../../
neuron=$1
threads=$2
conv_layer_id1=$3
conv_layer_id2=$4
config=$5
# neuron=64
# threads=4
# config=../configs/interpreting/saluki_model_multi_run.yaml

for ((x=$conv_layer_id1; x<=$conv_layer_id2; x++))
do
    task=adample
    bash multi_run_layer.sh \
    $x \
    $neuron \
    $threads \
    $config \
    $task 

    task=tomtom
    bash multi_run_layer.sh \
    $x \
    $neuron \
    $threads \
    $config \
    $task 

done

