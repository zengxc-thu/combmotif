#!/bin/bash
layer=$1
kernels=$2
threads=$3
conv_name="conv$layer"
config=$4
task=$5

bash idx.sh  $kernels  | xargs -n 1 -P $threads  bash   ${task}.sh $conv_name $config

