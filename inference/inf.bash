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

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=spawn


HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *ws* ]]; then

    ### if the machine name is ws006601 then set the parent path to  /home/abdelrahman.sadallah
    if [[ "$HOSTNAME" == "ws006601" ]]; then
        CHECKPOINT_PARENT_PATH="/home/abdelrahman.sadallah"
        PARENT_PATH="/home/abdelrahman.sadallah"
    else 
        CHECKPOINT_PARENT_PATH="/mnt/data/users/boda"
        PARENT_PATH="/mnt/data/users/boda"
    fi
    CHECKPOINT_PARENT_PATH="$CHECKPOINT_PARENT_PATH/review_rewrite_chekpoints"
    export CUDA_VISIBLE_DEVICES=0,1
    export TRITON_CACHE_DIR=$PARENT_PATH
    export HF_CACHE_DIR=$PARENT_PATH/huggingface
    export HF_HOME=$PARENT_PATH/huggingface
else
    CHECKPOINT_PARENT_PATH="/l/users/abdelrahman.sadallah/review_evaluation"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export TRITON_CACHE_DIR="/l/users/$USER/"
    export HF_CACHE_DIR="/l/users/$USER/hugging_face"
fi
echo "CHECKPOINT_PARENT_PATH: $CHECKPOINT_PARENT_PATH"

GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
export CUDA_LAUNCH_BLOCKING=1
echo "GPUS: $GPUS"

WRITE_PATH="evalute_outputs"


# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# "meta-llama/Llama-3.1-8B"
# "allenai/scitulu-7b"
# "Uni-SMART/SciLitLLM"

###### MODEL CONFIG ########
FULL_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GENERATION_TYPE="score_only"
PROMPT_TYPE="chat"
STEP="843"
FINETUNING_TYPE="adapters"
####### DATA CONFIG ########
ASPECT="all"
DATASET_SPLIT="test"
GOLD_LABEL_FORMAT="ASPECT_label"
DATASET_NAME="boda/review_evaluation_automatic_labels"


## Determine the write path based on the finetuning type
if [ "$FINETUNING_TYPE" == "adapters" ]; then
    WRITE_PATH="$WRITE_PATH/adapters"
    echo "Adapters Finetuned Model Evaluation"
elif [ "$FINETUNING_TYPE" == "full_finetuning" ]; then
    WRITE_PATH="$WRITE_PATH/full_model"
    echo "Full Finetuned Model Evaluation"
else
    WRITE_PATH="$WRITE_PATH/base_model"
    echo "Base Model Evaluation"
fi

python  vllm_inf.py \
--step $STEP \
--finetuning_type $FINETUNING_TYPE \
--checkpoint_parent_path $CHECKPOINT_PARENT_PATH \
--prompt_type $PROMPT_TYPE \
--generation_type $GENERATION_TYPE \
--full_model_name  $FULL_MODEL_NAME \
--tensor_parallel_size $GPUS \
--output_path $WRITE_PATH \
--max_new_tokens 64 \
--temperature 0.1 \
--dataset_config $ASPECT \
--dataset_split $DATASET_SPLIT \
--gold_label_format $GOLD_LABEL_FORMAT \
--dataset_name $DATASET_NAME \

