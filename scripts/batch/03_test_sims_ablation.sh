#!/bin/bash

num_sims=()
num_sims+=(1)
num_sims+=(2)
num_sims+=(4)
#num_sims+=(8)
#num_sims+=(16)
#num_sims+=(32)
#num_sims+=(64)
#num_sims+=(128)
#num_sims+=(256)
#num_sims+=(512)

folder='nn_ablation_128_5'
params="-nn_num_channels 256 -nn_lin_size 1024"

i=19
for minimax in $(seq 1 1 6); do
  for nsim in "${num_sims[@]}"
  do
    CUDA_VISIBLE_DEVICES= PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -white minimax"$minimax" -black data/$folder/training_$i.pth.tar -results_file data/$folder/progress_minimax"$minimax"_vs_nn"$nsim".csv --wait_for_nn -note iter_$i $params -mcts_num_sims $nsim &
    CUDA_VISIBLE_DEVICES= PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -black minimax"$minimax" -white data/$folder/training_$i.pth.tar -results_file data/$folder/progress_nn"$nsim"_vs_minimax"$minimax".csv --wait_for_nn -note iter_$i $params -mcts_num_sims $nsim &
  done
done
