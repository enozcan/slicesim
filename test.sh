#!/bin/bash

# $1 must be the config file.

CLIENT_COUNT=(5000 10000 15000 20000)
LB_TYPES=('disabled' 'mean' 'max')

for cc in "${CLIENT_COUNT[@]}"
do
  :
    for lb in "${LB_TYPES[@]}"
    do
      :
          export NUM_CLIENTS=$cc
	        export LOAD_BALANCE_TYPE=$lb
          export OUTPUT_FILE_NAME=test_results/$cc-$lb/$cc-$lb
          mkdir -p test_results/$cc-$lb
          python3 -m slicesim dummy $1
    done
done
