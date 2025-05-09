#!/bin/bash

# Find the absolute path of the directory containing this script
# SCRIPT_DIR will be like /path/to/baby-llm-infer/baby_llm_infer_v2
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Calculate the project root directory (one level up from SCRIPT_DIR)
# PROJECT_ROOT will be like /path/to/baby-llm-infer
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Change the current directory to the project root
# This ensures python -m can find the baby_llm_infer_v2 module
cd "$PROJECT_ROOT"

# Check if the cd was successful (optional but good practice)
if [ $? -ne 0 ]; then
  echo "Error: Failed to change directory to $PROJECT_ROOT"
  exit 1
fi

echo "Executing python command from: $(pwd)" # Add this line for debugging

# Single Request (Example, commented out)
# python -m baby_llm_infer_v2.main --model_name Qwen/Qwen3-1.7B \
# --prompt "The capital of China is" \
# --max_tokens 500 \
# --trust_remote_code \

# Execute the batch inference script
# python -m baby_llm_infer_v2.main --model_name Qwen/Qwen3-1.7B \
# --max_tokens 50 \
# --trust_remote_code \
# --batch

# Optional: Change back to the original directory if needed
# cd "$OLDPWD"

python -m baby_llm_infer_v2.main --model_name openai-community/gpt2-large \
--max_tokens 50 \
--trust_remote_code \
--batch