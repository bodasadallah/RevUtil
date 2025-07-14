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
export NCCL_P2P_LEVEL=NVL

HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *ws* ]]; then

    ### if the machine name is ws006601 then set the parent path to  /home/abdelrahman.sadallah
    if [[ "$HOSTNAME" == "ws006601" ]]; then
        CHECKPOINT_PARENT_PATH="/home/abdelrahman.sadallah"
        PARENT_PATH="/home/abdelrahman.sadallah"
    else 
        CHECKPOINT_PARENT_PATH="/mnt/data/users/$USER"
        PARENT_PATH="/mnt/data/users/$USER"
    fi
    CHECKPOINT_PARENT_PATH="$CHECKPOINT_PARENT_PATH/review_rewrite_chekpoints"
    export CUDA_VISIBLE_DEVICES=0,1
    export TRITON_CACHE_DIR=$PARENT_PATH/
    export HF_CACHE_DIR=$PARENT_PATH/huggingface
    export HF_HOME=$PARENT_PATH/huggingface

########################## CSCC ###########################
else
    CHECKPOINT_PARENT_PATH="/l/users/$USER/review_evaluation"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export TRITON_CACHE_DIR="/l/users/$USER/"
    export HF_CACHE_DIR="/l/users/$USER/hugging_face"
fi
echo "CHECKPOINT_PARENT_PATH: $CHECKPOINT_PARENT_PATH"

GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
export CUDA_LAUNCH_BLOCKING=1
echo "GPUS: $GPUS"

WRITE_PATH="evaluate_outputs"



# PROMPT_TYPE="chat"
STEP="0"
FINETUNING_TYPE="adapters"
MAX_NUM_SEQS=16
TRAINING_aspects="all"

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


# MODELS=("Uni-SMART/SciLitLLM" "WestlakeNLP/DeepReviewer-7B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "meta-llama/Llama-3.1-8B-Instruct" "allenai/scitulu-7b" "meta-llama/Llama-3.1-8B")
# DATASETS=("boda/review_evaluation_human_annotation" "boda/review_evaluation_human_annotation" "boda/review_evaluation_automatic_labels")
# DATASET_SPLITS=("gold,silver,hard,full" "full" "test")
# DATASET_CONFIGS=("actionability,grounding_specificity,verifiability,helpfulness" "combined_main_aspects" "all")

# DATASETS=("boda/review_evaluation_human_annotation"  "boda/review_evaluation_automatic_labels")
# DATASET_SPLITS=("full" "test")
# DATASET_CONFIGS=("combined_main_aspects" "all")

# GENERATION_TYPES=("score_only" "score_rationale")



# MODELS=( "prometheus-eval/prometheus-7b-v2.0" "chatgpt" "Uni-SMART/SciLitLLM" "WestlakeNLP/DeepReviewer-7B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "meta-llama/Llama-3.1-8B-Instruct" "allenai/scitulu-7b" "meta-llama/Llama-3.1-8B" )
# MODELS=("Uni-SMART/SciLitLLM" "WestlakeNLP/DeepReviewer-7B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "meta-llama/Llama-3.1-8B-Instruct" "allenai/scitulu-7b" "meta-llama/Llama-3.1-8B" )

# "meta-llama/Llama-3.2-3B-Instruct"
#  "meta-llama/Llama-3.1-8B-Instruct" 
MODELS=( "meta-llama/Llama-3.2-3B-Instruct" "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

GENERATION_TYPES=("score_only")
DATASETS=("boda/review_evaluation_human_annotation"  "boda/review_evaluation_automatic_labels")
DATASET_SPLITS=("full" "test")
DATASET_CONFIGS=("combined_main_aspects" "all")




for MODEL in "${MODELS[@]}"; do
    for GENERATION_TYPE in "${GENERATION_TYPES[@]}"; do
        for i in "${!DATASETS[@]}"; do
        

            # for S in $(seq 144 -16 16); do
            for S in 0; do

                STEP=$S

                if [ "$STEP" -eq 144 ] && [[ "$MODEL" == *"SciLitLLM"* ]]; then
                    STEP=141
                fi
                echo "Processing Step: $STEP"


                DATASET_NAME=${DATASETS[$i]}
                DATASET_SPLIT=${DATASET_SPLITS[$i]}
                ASPECT=${DATASET_CONFIGS[$i]}

                if [ "$FINETUNING_TYPE" == "baseline" ]; then
                    PROMPT_TYPE="chat"
                elif [ "$FINETUNING_TYPE" == "adapters" ]; then
                    PROMPT_TYPE="instruction"
                fi


                ### if the model name is "meta-llama/Llama-3.1-8B" then make prompt type "instruction" don't do that for "meta-llama/Llama-3.1-8B-Instruct"
                ### or the model is chatgpt
                if [[ (("$MODEL" == *"Llama-3.1-8B"* && "$MODEL" != *"Instruct"*) || "$MODEL" == "chatgpt" || "$MODEL" == *"prometheus"*) && "$MODEL" != *"Selene"* ]]; then
                    PROMPT_TYPE="instruction"
                fi
                echo "PROMPT_TYPE: $PROMPT_TYPE"
                echo "MODEL: $MODEL"
                echo "DATASET_NAME: $DATASET_NAME"
                echo "DATASET_SPLIT: $DATASET_SPLIT"
                echo "ASPECT: $ASPECT"
                echo "GENERATION_TYPE: $GENERATION_TYPE"

                ### if dataset name have automatic labels then the gold label format should be "chatgpt_ASPECT_score", else "ASPECT_label"
                if [[ "$DATASET_NAME" == *"automatic"* ]]; then
                    GOLD_LABEL_FORMAT="chatgpt_ASPECT_score"
                else
                    # GOLD_LABEL_FORMAT="ASPECT_label"
                    GOLD_LABEL_FORMAT="ASPECT"
                fi
                python  vllm_inf.py \
                --step $STEP \
                --max_num_seqs $MAX_NUM_SEQS \
                --training_aspects $TRAINING_aspects \
                --finetuning_type $FINETUNING_TYPE \
                --checkpoint_parent_path $CHECKPOINT_PARENT_PATH \
                --prompt_type $PROMPT_TYPE \
                --generation_type $GENERATION_TYPE \
                --full_model_name  $MODEL \
                --tensor_parallel_size $GPUS \
                --output_path $WRITE_PATH \
                --max_new_tokens 1024 \
                --temperature 0 \
                --dataset_config $ASPECT \
                --dataset_split $DATASET_SPLIT \
                --gold_label_format $GOLD_LABEL_FORMAT \
                --dataset_name $DATASET_NAME \

            done
        done
    done
done

