#!/bin/bash

# update according to your python path
PYTHON="python3"

model_file_arr=( "None" "None" "None" "None" )

num_features_arr=( 32 32 32 32 )

num_epochs_arr=( 5000 5000 5000 5000 )

mil_filter_array=( "distribution" "mean" "attention" "max" )

count=0
for MIL_FILTER in "${mil_filter_array[@]}"
do

        echo "MIL_FILTER="$MIL_FILTER

        MODEL_FILE=${model_file_arr[$count]}

        NUM_FEATURES=${num_features_arr[$count]}

        NUM_EPOCHS=${num_epochs_arr[$count]}

        $PYTHON train.py --mil_pooling_filter $MIL_FILTER --init_model_file $MODEL_FILE --num_features $NUM_FEATURES --num_epochs $NUM_EPOCHS --save_interval 20

        count=$((count + 1))

        # exit 1

done