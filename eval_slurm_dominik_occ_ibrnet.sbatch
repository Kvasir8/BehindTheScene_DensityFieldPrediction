#!/bin/bash
#SBATCH --job-name="eval_occ"
#SBATCH --partition=DEADLINE
#SBATCH --comment="CVPR"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/hank/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/user/hank/slurm/logs/slurm-%j.out
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
pwd; hostname; date
nvidia-smi

configfile="eval_lidar_occ_dominik_ibrnet"
encoder_ids="[0,1,2,3]" # mono 

python eval.py -cn $configfile ++model_conf.encoder_ids=$encoder_ids
