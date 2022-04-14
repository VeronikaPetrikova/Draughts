#!/bin/bash


nsim=256
folder=j
for i in $(seq 0 1 1000); do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv --wait_for_nn -note iter_$i -nn_lin_size 4096 -nn_num_channels 1024
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 25 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv --wait_for_nn -note iter_$i -nn_lin_size 4096 -nn_num_channels 1024
done


#nsim=25
#folder=e
#for i in $(seq 0 1 1000); do
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#done
#nsim=50
#folder=e
#for i in $(seq 0 1 1000); do
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#done

#nsim=25
#folder=d
#for i in $(seq 0 1 30); do
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#done
#nsim=100
#folder=d
#for i in $(seq 0 1 30); do
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#done
#
#nsim=50
#folder=c
#for i in $(seq 0 1 30); do
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#done
#nsim=100
#folder=c
#for i in $(seq 0 1 30); do
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -white minimax2 -black data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_minimax2_vs_nn"$nsim".csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:`pwd`" python scripts/compare.py 100 -black minimax2 -white data/$folder/training_$i.pth.tar -mcts_num_sims $nsim -results_file data/$folder/progress_nn"$nsim"_vs_minimax2.csv -note iter_$i -nn_lin_size 2048 -game_max_moves 75 -game_almost_win_value 0
#done
