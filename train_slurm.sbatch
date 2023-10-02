#!/bin/bash
#SBATCH --job-name="longrun_DFTunfr_do01_DFTfixed_ff_encstyle_default_enly6_nv8_att16_nh4_feOffset_pgt_lr1e4_rbs6144"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/hank/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/user/hank/slurm/logs/slurm-%j.out
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
pwd; hostname; date
nvidia-smi
python train.py -cn exp_kitti_360_DFT_slurm

#export MY_PORT=58024  # Define the desired port number as an environment variable
