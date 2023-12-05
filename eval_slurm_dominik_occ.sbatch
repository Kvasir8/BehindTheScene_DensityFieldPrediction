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

# configfile="dominik_eval_lidar_occ_feature_fusion"
# checkpoint="/storage/user/muhled/outputs/mvbts/kitti-360/simple_multi_view_head_alt2_backend-None-1_20231101-095135/training_checkpoint_150000.pt"
configfile="dominik_eval_lidar_occ_feature_fusion"
checkpoint="/storage/user/muhled/outputs/mvbts/kitti-360/BaseExperimentNoFisheyeFeatureFusionLarge_backend-None-1_20231107-105543/training_checkpoint_200000.pt"
# checkpoint="/storage/user/muhled/outputs/mvbts/kitti-360/BaseExperimentNoFisheyeFeatureFusionLargeKnowDistResume_backend-None-1_20231111-131526/training_checkpoint_250000.pt"
new_architecture="true"
# no spaces are important
encoder_ids="[0]" # mono 
# encoder_ids="[0,1]" # temporal 
# encoder_ids="[0,2]" # stereo 
# encoder_ids="[0,1,2,3]" # stereo-temporal 
# encoder_ids="[0,1,2,3,4,5,6,7]" # full 

python eval.py -cn $configfile ++model_conf.encoder_ids=$encoder_ids ++checkpoint=$checkpoint ++new_model_architecture=$new_architecture
