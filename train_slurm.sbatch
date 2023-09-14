#!/bin/bash
#SBATCH --job-name="ff_nmv4_att16_lay4_enc0_do0_Unfr_B1_lr5e5_BTS_Nry_woViedir_feFalse_unbiased"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
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
python train.py -cn exp_kitti_360_DFT

#export MY_PORT=58024  # Define the desired port number as an environment variable
