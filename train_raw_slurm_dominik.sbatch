#!/bin/bash
#SBATCH --job-name="dominik_kitti_raw_feature_fusion_small"
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="CVPR"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/hank/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/user/hank/slurm/logs/slurm-%j.out
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
pwd; hostname; date
nvidia-smi

python train.py -cn dominik_kitti_raw_feature_fusion_small
