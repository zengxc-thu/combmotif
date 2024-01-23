#!/bin/bash

python ../tfmodisco_lite.py --name ${1} \
--config $2 \
--ind ${3}-$(( $3 + 1)) 


