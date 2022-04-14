#!/bin/bash

for i in $(seq 1 1 8); do
  for j in $(seq 1 1 8); do
    if [ $i == $j ]; then
        continue
    fi
    echo "$i vs $j"
    PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -white minimax$i -black minimax$j -results_file data/minimax"$i"_vs_minimax"$j".csv -note miminax_"$i"_vs_"$j" &
  done
done