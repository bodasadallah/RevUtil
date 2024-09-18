#!/bin/bash

#SBATCH --job-name=test_cscc
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=48G                   # Total RAM to be used
#SBATCH --cpus-per-task=64        # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH -p cscc-gpu-p                     # Use the gpu partition
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs



echo "starting Inf......................."

cd inference
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
# CUDA_VISIBLE_DEVICES=3
echo $CUDA_VISIBLE_DEVICES
echo $CONDA_PREFIX
python vllm_inf.py \
--model_name  'gpt-4o-mini' \
--input_path '../data/annotated_review_points_batch_2.csv' \
--output_path '../outputs/boda_gpt4_output_batch_2.csv' \
--batch_size 16 \
--total_points 0 \
--run_statistics 1 \
--compare_to_human 1 \
--statistics_path '/home/abdelrahman.sadallah/mbzuai/review_rewrite/outputs/stats_gpt4_batch_2.txt' 
# --aspect 'actionability'

# 'gpt-4o-mini'
# 'google/gemma-2-9b-it'

