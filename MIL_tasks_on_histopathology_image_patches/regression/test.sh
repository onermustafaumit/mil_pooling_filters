#!/bin/bash

# update according to your python path
PYTHON="python3"

model_file_arr=( "2020_12_04__13_09_06__4920" "2020_05_24__08_01_28__3160" "2020_05_25__12_41_00__4000" "2020_05_25__12_41_45__4000" )

num_features_arr=( 32 32 32 32 )

mil_filter_array=( "distribution" "mean" "attention" "max" )

count=0
for MIL_FILTER in "${mil_filter_array[@]}"
do

	echo "MIL_FILTER="$MIL_FILTER

	MODEL_FILE="saved_models/state_dict__"${model_file_arr[$count]}".pth"
	echo "MODEL_FILE="$MODEL_FILE

	NUM_FEATURES=${num_features_arr[$count]}

	$PYTHON test.py --mil_pooling_filter $MIL_FILTER --init_model_file $MODEL_FILE --num_features $NUM_FEATURES
	
	DATA_FOLDER_PATH="test_metrics/"${model_file_arr[$count]}"/test"
	echo "DATA_FOLDER_PATH="$DATA_FOLDER_PATH

	$PYTHON collect_statistics_over_bag_predictions.py --data_folder_path $DATA_FOLDER_PATH
	
	count=$((count + 1))

	# exit 1

done

$PYTHON calculate_paired_sample_t_test.py

