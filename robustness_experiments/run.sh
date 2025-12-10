#!/bin/bash

# RevUtil Robustness Experiments Runner
# This script provides an easy interface to run the robustness experiments

set -e  # Exit on error
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "=========================================="
echo "  RevUtil Robustness Experiments"
echo "=========================================="
echo -e "${NC}"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if vLLM is installed
if ! python -c "import vllm" &> /dev/null; then
    echo -e "${YELLOW}Warning: vLLM is not installed. Installing requirements...${NC}"
    pip install vllm pandas tqdm
fi

# Parse arguments
EXPERIMENT="all"
GPU="1"

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --experiment TYPE    Experiment to run: typo, jailbreak, no_system_prompt,"
            echo "                           score_manipulation, score_manipulation_paired, or all (default: all)"
            echo "  -g, --gpu ID            GPU ID to use (default: 1)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Run all experiments on GPU 1"
            echo "  $0 -e typo                                 # Run only typo test"
            echo "  $0 -e score_manipulation_paired            # Run paired manipulation comparison"
            echo "  $0 -e all -g 0                             # Run all experiments on GPU 0"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU
echo -e "${GREEN}Using GPU: $GPU${NC}"
echo ""

# Check if test sets exist
if [ ! -d "test_sets" ]; then
    echo -e "${RED}Error: test_sets directory not found${NC}"
    exit 1
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Display experiment info
echo -e "${BLUE}Running experiment(s): ${EXPERIMENT}${NC}"
echo ""

# Run the Python script
python run_experiments.py --experiment "$EXPERIMENT"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "  Experiments completed successfully!"
    echo -e "==========================================${NC}"
    echo ""
    echo -e "Results saved in: ${BLUE}outputs/${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Analyze results: ${YELLOW}jupyter notebook analyze_results.ipynb${NC}"
    echo -e "  2. View raw outputs: ${YELLOW}ls -lh outputs/${NC}"
    echo -e "  3. Check CSV files: ${YELLOW}cat outputs/*_results_*.csv${NC}"
else
    echo ""
    echo -e "${RED}=========================================="
    echo "  Experiments failed!"
    echo -e "==========================================${NC}"
    echo ""
    echo "Check the error messages above for details."
    exit 1
fi
