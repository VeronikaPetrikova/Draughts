#!/bin/bash


#nsim=25
#folder=f
#for i in $(seq 0 1 40); do
#  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 4096 -nn_num_channels 1024 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 4096 -nn_num_channels 1024 -game_max_moves 75 -game_almost_win_value 0
#done
#nsim=50
#folder=f
#for i in $(seq 0 1 40); do
#  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 4096 -nn_num_channels 1024 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 4096 -nn_num_channels 1024 -game_max_moves 75 -game_almost_win_value 0
#done
nsim=100
folder=f
for i in $(seq 3 5 40); do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 4096 -nn_num_channels 1024 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 4096 -nn_num_channels 1024 -game_max_moves 75 -game_almost_win_value 0
done

