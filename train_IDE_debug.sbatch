#!/bin/bash
#SBATCH --job-name="[IDE]BTS_DFT"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=48G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/hank/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/user/hank/slurm/logs/slurm-%j.out
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
pwd; hostname; date
nvidia-smi
python train_pycharm.py -cn exp_kitti_360_DFT ## -nproc_per_node 2

#export MY_PORT=58024  # Define the desired port number as an environment variable