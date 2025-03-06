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

export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
export CUDA_LAUNCH_BLOCKING=1
echo "GPUS: $GPUS"
# FULL_MODEL_NAME="/l/users/abdelrahman.sadallah/review_evaluation/full_training/all/checkpoint-109"

# "meta-llama/Llama-3.1-8B"
# "allenai/scitulu-7b"
# "Uni-SMART/SciLitLLM"

FULL_MODEL_NAME="/l/users/abdelrahman.sadallah/review_evaluation/full/SciLitLLM/score_only/instruction/all/checkpoint-268"
# FINE_TUNED_MODEL_PATH="/l/users/abdelrahman.sadallah/review_evaluation/adapters/scitulu-7b/score_only/instruction/all"


GENERATION_TYPE="score_only"
ASPECT="all"
WRITE_PATH="evalute_outputs"
PROMPT_TYPE="instruction"



## If FINE_TUNED_MODEL_PATH is not set, then set the path to the full model1
if [ -z "$FINE_TUNED_MODEL_PATH" ]; then
    WRITE_PATH="$WRITE_PATH/full_model"
    echo "FINE_TUNED_MODEL_PATH is not set, using full model"
else
    WRITE_PATH="$WRITE_PATH/adapters"
    echo "FINE_TUNED_MODEL_PATH is set, using adapters"
fi

python  vllm_inf.py \
--prompt_type $PROMPT_TYPE \
--generation_type $GENERATION_TYPE \
--base_model_name  $FULL_MODEL_NAME \
--tensor_parallel_size $GPUS \
--output_path $WRITE_PATH \
--max_new_tokens 64 \
--temperature 0.1 \
--dataset_config $ASPECT \
--dataset_split "gold" \
--gold_label_format "ASPECT_label" \
--dataset_name "boda/review_evaluation_automatic_labels" \
# --finetune_model_name $FINE_TUNED_MODEL_PATH \
