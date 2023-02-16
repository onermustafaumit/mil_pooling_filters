#!/bin/bash

# update according to your python path
PYTHON="python3"

echo "collect_acc_values.py"
$PYTHON collect_acc_values.py

echo "plot_color_coded_p_value_maps.py"
$PYTHON plot_color_coded_p_value_maps.py

