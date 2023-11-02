#!/bin/bash
#SBATCH --job-name="RFull_eval_depth_MVBTS_full"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
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
python eval.py -cn eval_depth_MVBTS
# python eval.py -cn eval_depth_MVBTS_full
# python eval.py -cn eval_depth_MVBTS_singleview

## Occupancy evaluation
# python eval.py -cn eval_lidar_occ_MVBTS_NoSF
# python eval.py -cn eval_lidar_occ_MVBTS_NoF
# python eval.py -cn eval_lidar_occ_MVBTS_full

# python eval.py -cn eval_lidar_occ_KDBTS_full


#export MY_PORT=58024  # Define the desired port number as an environment variable