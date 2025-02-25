#!/bin/bash

#SBATCH --job-name=NLP804 # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=256GB                  # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --qos=gpu-8 
#SBATCH -p gpu                      # Use the gpu partition
##SBATCH --nodelist=ws-l6-017


# ###### CSCC PATHS ######
# export TRITON_CACHE_DIR="/l/users/$USER/"
# export HF_CACHE_DIR="/l/users/$USER/hugging_face"

export TRITON_CACHE_DIR="/home/$USER/"
export HF_CACHE_DIR="/home/$USER/hugging_face"
export CUDA_VISIBLE_DEVICES=0,1

# get the number of GPUs and store them into variable
GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPUS: $GPUS"

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file deepspeed_zero3.yaml --num_processes=$GPUS \
run_sft.py \
config_full.yaml

# accelerate launch \
# --config_file deepspeed_zero3.yaml \
# run_sft.py \