#!/bin/bash

num_sims=()
#num_sims+=(1)
#num_sims+=(2)
#num_sims+=(4)
#num_sims+=(8)
#num_sims+=(16)
#num_sims+=(32)
#num_sims+=(64)
num_sims+=(128)
#num_sims+=(256)
#num_sims+=(512)

cpucts=()
cpucts+=('0.5')
cpucts+=('0.1')
cpucts+=('0.01')
cpucts+=('0.001')
cpucts+=('2')
cpucts+=('5')
cpucts+=('10')

folder='nn_ablation_128_5'
params="-nn_num_channels 256 -nn_lin_size 1024"
minimax=4
i=19
for nsim in "${num_sims[@]}"
do
  for cpuct in "${cpucts[@]}"
    do
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -white minimax"$minimax" -black data/$folder/training_$i.pth.tar -results_file data/$folder/cpuct2_minimax"$minimax"_vs_nn"$nsim".csv --wait_for_nn -note "cpuct_$cpuct" $params -mcts_num_sims $nsim -mcts_cpuct $cpuct
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -black minimax"$minimax" -white data/$folder/training_$i.pth.tar -results_file data/$folder/cpuct2_nn"$nsim"_vs_minimax"$minimax".csv --wait_for_nn -note "cpuct_$cpuct" $params -mcts_num_sims $nsim -mcts_cpuct $cpuct
  done
done
