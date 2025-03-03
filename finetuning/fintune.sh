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
export TRITON_CACHE_DIR="/l/users/$USER/"
export HF_CACHE_DIR="/l/users/$USER/hugging_face"

########## NLP WORKSTATION PATHS ##########
# export TRITON_CACHE_DIR="/home/$USER/"
# export HF_CACHE_DIR="/home/$USER/hugging_face"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# get the number of GPUs and store them into variable
GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPUS: $GPUS"


##################### TRAINING CONF ################
# Get the hostname
HOSTNAME=$(hostname)
# Check if "ws" is in the hostname
if [[ "$HOSTNAME" == *ws* ]]; then
    OUTPUTPATH="../review_evaluation_checkpoints"
else
    OUTPUTPATH="/l/users/abdelrahman.sadallah/review_evaluation"  # You can change this default if needed
fi




## Use adapter or full model tuning
USE_PEFT=false
### 

## if USE_PEFT is true, then append "adapters" to the output path
if [ "$USE_PEFT" = true ]; then
    OUTPUTPATH="$OUTPUTPATH/adapters"
    echo "Using adapters"
else
    OUTPUTPATH="$OUTPUTPATH/full"
    echo "Using full model"
fi

echo "OUTPUTPATH is set to: $OUTPUTPATH"


ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file deepspeed_zero3.yaml --num_processes=$GPUS \
run_sft.py \
config_full.yaml \
--output_dir=$OUTPUTPATH \
--use_peft=$USE_PEFT 


# --config_file fsdp.yaml \
# --config_file multi_gpu.yaml \
