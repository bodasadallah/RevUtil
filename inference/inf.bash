#!/bin/bash

#SBATCH --job-name=test_cscc
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=48G                   # Total RAM to be used
#SBATCH --cpus-per-task=64        # Number of CPU cores
#SBATCH --time=12:00:00   

#SBATCH -p cscc-cpu-p                     
#SBATCH -q cscc-cpu-qos 
         
##SBATCH --gres=gpu:1                # Number of GPUs (per node)
##SBATCH -p cscc-gpu-p                     
##SBATCH -q cscc-gpu-qos 



echo "starting Inf......................."

# cd inference
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
echo $CUDA_VISIBLE_DEVICES
echo $CONDA_PREFIX
# CUDA_VISIBLE_DEVICES=3


python  vllm_inf.py \
--model_name  "$FULL_MODEL_NAME" \
--output_path "evalute_outputs" \
--max_new_tokens 64 \
--temperature 0.1 \
--finetune_model_name "/l/users/abdelrahman.sadallah/review_evaluation/actionability/checkpoint-33/" \
--dataset_config "actionability" \
--dataset_name "boda/review_evaluation_fine_tuning" \