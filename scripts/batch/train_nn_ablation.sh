#!/bin/bash
#
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_0 -nn_num_channels 256 -nn_lin_size 1024
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_1 -nn_num_channels 512 -nn_lin_size 2048
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_2 -nn_num_channels 1024 -nn_lin_size 4084


#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_128_0 -mcts_num_sims 128 -num_self_play 256 -nn_num_channels 64 -nn_lin_size 256
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_128_1 -mcts_num_sims 128 -num_self_play 256 -nn_num_channels 128 -nn_lin_size 512
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_128_2 -mcts_num_sims 128 -num_self_play 256 -nn_num_channels 32 -nn_lin_size 128
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_128_3 -mcts_num_sims 128 -num_self_play 256 -nn_num_channels 16 -nn_lin_size 64
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_128_4 -mcts_num_sims 128 -num_self_play 256 -nn_num_channels 8 -nn_lin_size 32


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_128_5 -mcts_num_sims 128 -num_self_play 256 -nn_num_channels 256 -nn_lin_size 1024
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_128_6 -mcts_num_sims 128 -num_self_play 256 -nn_num_channels 512 -nn_lin_size 2048
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:`pwd` python scripts/train.py -folder nn_ablation_128_7 -mcts_num_sims 128 -num_self_play 256 -nn_num_channels 1024 -nn_lin_size 4084