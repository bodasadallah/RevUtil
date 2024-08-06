#!/bin/bash

#SBATCH --job-name=gemma-9b-inf # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p it-hpc                     # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment


echo "starting Inf......................."

cd notebooks/inference

python vllm_inf.py