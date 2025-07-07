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


export NCCL_P2P_LEVEL=NVL
export WANDB_LOG_MODEL=false
##################### TRAINING CONF ################
# Get the hostname
HOSTNAME=$(hostname)
# Check if "ws" is in the hostname
if [[ "$HOSTNAME" == *ws* ]]; then

    ### if the machine name is ws006601 then set the parent path to  /home/abdelrahman.sadallah
    if [[ "$HOSTNAME" == "ws006601" ]]; then
        PARENT_PATH="/home/abdelrahman.sadallah"
    else 
        PARENT_PATH="/mnt/data/users/$USER"
    fi

    OUTPUTPATH="$PARENT_PATH/review_rewrite_chekpoints"  # You can change this default if needed
    
    ## if the output path don't exist, create it
    if [ ! -d "$OUTPUTPATH" ]; then
        mkdir -p $OUTPUTPATH
    fi
    export TRITON_CACHE_DIR=$PARENT_PATH
    export HF_CACHE_DIR=$PARENT_PATH/huggingface
    export HF_HOME=$PARENT_PATH/huggingface
    export CUDA_VISIBLE_DEVICES=0,1

########################## CSCC ###########################
else
    OUTPUTPATH="/l/users/abdelrahman.sadallah/review_evaluation"  # You can change this default if needed
        ## if the output path don't exist, create it
    if [ ! -d "$OUTPUTPATH" ]; then
        mkdir -p $OUTPUTPATH
    fi
    export TRITON_CACHE_DIR="/l/users/$USER/"
    export HF_CACHE_DIR="/l/users/$USER/hugging_face"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi



export CUDA_LAUNCH_BLOCKING=1
# get the number of GPUs and store them into variable
export GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPUS: $GPUS"

## Use adapter or full model tuning
USE_PEFT=true
GENERATION_TYPE="score_only"
# ASPECTS=(  "actionability" "grounding_specificity" "verifiability" "helpfulness" )
ASPECTS=("all")


# List of models to iterate over
MODELS=(  "meta-llama/Meta-Llama-3-70B-Instruct")

for A in "${ASPECTS[@]}"; do

    ASPECT=$A
    echo "Processing Aspect: $ASPECT"
    for MODEL_NAME_OR_PATH in "${MODELS[@]}"; do
        echo "Finetuning Model: $MODEL_NAME_OR_PATH"

        ## if USE_PEFT is true, then append "adapters" to the output path
        if [ "$USE_PEFT" = true ]; then
            MODEL_OUTPUTPATH="$OUTPUTPATH/adapters/"
            echo "Using adapters"
            echo "MODEL_OUTPUTPATH: $MODEL_OUTPUTPATH"
        else
            MODEL_OUTPUTPATH="$OUTPUTPATH/full/"
            echo "Using full model"
            echo "MODEL_OUTPUTPATH: $MODEL_OUTPUTPATH"
        fi

        ## if the model output path doesn't exist, create it
        if [ ! -d "$MODEL_OUTPUTPATH" ]; then
            mkdir -p $MODEL_OUTPUTPATH
        fi

        echo "OUTPUTPATH is set to: $MODEL_OUTPUTPATH"

        ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file deepspeed_zero3.yaml --num_processes=$GPUS \
        run_sft.py \
        config_full.yaml \
        --output_dir=$MODEL_OUTPUTPATH \
        --use_peft=$USE_PEFT \
        --aspect=$ASPECT \
        --model_name_or_path=$MODEL_NAME_OR_PATH \
        --generation_type=$GENERATION_TYPE 
    done
done
