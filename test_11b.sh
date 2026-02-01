#!/usr/bin/env bash

export HF_HOME="/data2/users/zhangjy/test_vlm/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_ENABLE_PARALLEL_LOADING="true"
export HF_TOKEN="$HF_TOKEN"

export CUDA_VISIBLE_DEVICES=1,2,3,4,5

mkdir -p ./eval_results

# Use local path directly to skip download check
export MODEL_PATH="/data2/users/zhangjy/test_vlm/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"

accelerate launch --num_processes=5 --main_process_port 12399 -m lmms_eval \
  --model llama_vision \
  --model_args "pretrained=$MODEL_PATH,dtype=bfloat16" \
  --tasks mmmu_val \
  --batch_size 1 \
  --num_fewshot 0 \
  --output_path "./eval_results" \
  --log_samples