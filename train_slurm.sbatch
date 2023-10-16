#!/bin/bash
#SBATCH --job-name="360_encLy1_slurm"
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

### For ablation study
## kitti-360
python train.py -cn ab_kitti_360_encLy1
# python train.py -cn ab_kitti_360_encLy6
# python train.py -cn ab_kitti_360_dropoutZero
# python train.py -cn ab_kitti_360_dropout05
# python train.py -cn ab_kitti_360_embTypeff
# python train.py -cn ab_kitti_360_embdim16
# python train.py -cn ab_kitti_360_embdim64
## For message: Please change job-name to correspodingly e.g. if executing -cn ab_kitti_360_encLy1, then change job name as "360_encLy1_slurm"

## kitti-raw
# python train.py -cn ab_kitti_raw_encLy1
# python train.py -cn ab_kitti_raw_encLy6
# python train.py -cn ab_kitti_raw_dropoutZero
# python train.py -cn ab_kitti_raw_dropout05
# python train.py -cn ab_kitti_raw_embTypeff
# python train.py -cn ab_kitti_raw_embdim16
# python train.py -cn ab_kitti_raw_embdim64

# python train.py -cn exp_kitti_360_DFT_slurm
# python train.py -cn ab_kitti_raw_base

#export MY_PORT=58024  # Define the desired port number as an environment variable
