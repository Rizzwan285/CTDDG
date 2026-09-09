#!/bin/bash

NODES=(node004 node003 node001 node002)

for NODE in "${NODES[@]}"; do

    STATE=$(sinfo -h -n "$NODE" -o "%T")

    if [[ "$STATE" == "idle" ]]; then

        echo "========================================"
        echo "Selected node: $NODE"
        echo "========================================"

        if [[ "$NODE" == "node003" ]]; then
            PARTITION="gpu03"
            TIME="4-23:59:59"
        elif [[ "$NODE" == "node004" ]]; then
            PARTITION="gpu04"
            TIME="24:00:00"
        elif [[ "$NODE" == "node001" ]]; then
            PARTITION="gpu01"
            TIME="24:00:00"
        else
            PARTITION="gpu02"
            TIME="24:00:00"
        fi

        echo "Partition: $PARTITION"
        echo "Requested time: $TIME"
        echo "Submitting Jupyter job..."

        sbatch \
            --partition="$PARTITION" \
            --nodelist="$NODE" \
            --time="$TIME" \
            cluster/launch_jupyter.sh

        exit 0
    fi

    echo "$NODE is $STATE — skipping..."
done

echo "No preferred GPU node is currently idle."
exit 1
