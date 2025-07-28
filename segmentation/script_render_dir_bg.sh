#!/bin/bash

DATASET_PATH="/mnt/lustre/work/ponsmoll/pba534/ffgarments_datagen/data/4D-DRESS/"
START_INDEX=0
END_INDEX=81

for i in $(seq $START_INDEX $END_INDEX); do
    echo "Running set_bg2transp.py for index $i"
    python set_bg2transp.py "$DATASET_PATH" "$i"
done

echo "All processing complete."