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

# 'prometheus-eval/prometheus-7b-v2.0'
# 'gpt-4o-mini'
# 'google/gemma-2-9b-it'
# 'gpt-4o'
# 'gpt-4o-2024-08-06'

FULL_MODEL_NAME='gpt-4o-2024-08-06'
MODEL_NAME=${FULL_MODEL_NAME}


# Only the second half of the path for Prometheus
if [[ "$MODEL_NAME" == */* ]]; then
    MODEL_NAME=$(echo "$MODEL_NAME" | awk -F'/' '{print $2}')
fi
MAIN_DIR="outputs/${MODEL_NAME}"

echo "Model Name: ${MODEL_NAME}"
echo "Main Dir: ${MAIN_DIR}"

python   inference/vllm_inf.py \
--model_name  "$FULL_MODEL_NAME" \
--input_path 'data/manually_annotated_review_points.csv' \
--output_dir "$MAIN_DIR" \
--batch_size 1 \
--total_points 100 \
--max_new_tokens 1024 \
--run_statistics 1 \
--compare_to_human 1 \
--chatgpt_key "GPT4_KEY" \
--prompt "ternary_score_prompt" \
--aspect 'politeness'

# 'actionability'
# --statistics_path '$MAIN_DIR/stats_gpt4_batch_2.txt' 
# --input_path 'data/annotated_review_points_batch_2.csv' \
# ternary_score_prompt
