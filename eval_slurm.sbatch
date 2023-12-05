#!/bin/bash
#SBATCH --job-name="gen_imgs_Raw"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/hank/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/user/hank/slurm/logs/slurm-%j.out
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
pwd; hostname; date
nvidia-smi

## Depth evaluation
# python eval.py -cn eval_depth_MVBTS
# python eval.py -cn eval_depth_MVBTS_full
# python eval.py -cn eval_depth_MVBTS_singleview

# python eval.py -cn eval_depth_MVBTS

python scripts/images/gen_imgs.py

## Occupancy evaluation
# python eval.py -cn eval_lidar_occ
# python eval.py -cn dominik_eval_lidar_occ_final_model

# python eval.py -cn dominik_eval_lidar_occ_attn_layers_stereo_temp
# python eval.py -cn dominik_eval_lidar_occ_attn_layers_dominik

#export MY_PORT=58024  # Define the desired port number as an environment variable