#!/bin/bash
python ../neuronMotif_adaptive_sample_cluster.py --name ${1} \
--config $2 \
--ind ${3}-$(( $3 + 1)) 
