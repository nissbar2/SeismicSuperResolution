#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=16:0:0

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE="/cs/labs/werman/leeyam/sesmic/outputs/output_${TIMESTAMP}.log"
ERROR_FILE="/cs/labs/werman/leeyam/sesmic/outputs/error_${TIMESTAMP}.log"

source /cs/labs/werman/leeyam/virtualEnv/myenv/bin/activate

python /cs/labs/werman/leeyam/sesmic/main.py \
     > "$OUTPUT_FILE" 2> "$ERROR_FILE"
