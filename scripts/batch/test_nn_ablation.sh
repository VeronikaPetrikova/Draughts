#!/bin/bash

folders=()
folders+=("nn_ablation_128_0")
folders+=("nn_ablation_128_1")
folders+=("nn_ablation_128_2")
folders+=("nn_ablation_128_3")
folders+=("nn_ablation_128_4")
folders+=("nn_ablation_128_5")
folders+=("nn_ablation_128_6")
folders+=("nn_ablation_128_7")
all_params=()
all_params+=("-mcts_num_sims 128 -nn_num_channels 64 -nn_lin_size 256")
all_params+=("-mcts_num_sims 128 -nn_num_channels 128 -nn_lin_size 512")
all_params+=("-mcts_num_sims 128 -nn_num_channels 32 -nn_lin_size 128")
all_params+=("-mcts_num_sims 128 -nn_num_channels 16 -nn_lin_size 64")
all_params+=("-mcts_num_sims 128 -nn_num_channels 8 -nn_lin_size 32")
all_params+=("-mcts_num_sims 128 -nn_num_channels 256 -nn_lin_size 1024")
all_params+=("-mcts_num_sims 128 -nn_num_channels 512 -nn_lin_size 2048")
all_params+=("-mcts_num_sims 128 -nn_num_channels 1024 -nn_lin_size 4084")

minimax=4
for j in $(seq 4 1 8); do
  folder=${folders[j]}
  params=${all_params[j]}
  for i in $(seq 0 1 19); do
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -white minimax"$minimax" -black data/$folder/training_$i.pth.tar -results_file data/$folder/progress_minimax"$minimax"_vs_nn"$nsim".csv --wait_for_nn -note iter_$i $params
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -black minimax"$minimax" -white data/$folder/training_$i.pth.tar -results_file data/$folder/progress_nn"$nsim"_vs_minimax"$minimax".csv --wait_for_nn -note iter_$i $params
  done
done