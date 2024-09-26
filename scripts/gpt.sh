#!/bin/bash

GPU="0"
FOLDER="results_gpt"
CHOSEN_FILE="./results_gpt/tldr_gpt_SFT_validation_beam.json"
REJECTED_FILE="./results_gpt/tldr_gpt_SFT_validation_temp5.json"
BASE_MODEL="CarperAI/openai_summarize_tldr_sft"
TARGET_MODEL="tldr_gpt_MPO"
DECODING_TYPE="beam"
SPLIT="test"
DATASET="CarperAI/openai_summarize_tldr"
TOKENIZER_MODEL="EleutherAI/gpt-j-6b"
PATH_TO_ALIGNSCORE_CKPT=""

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./mpo.py \
    --rejected_file="${REJECTED_FILE}" \
    --chosen_file="${CHOSEN_FILE}" \
    --model_name_or_path="${BASE_MODEL}" \
    --output_dir="models_gpt/${TARGET_MODEL}" \
    --beta=0.5 \
    --learning_rate=1e-4 \
    --warmup_steps=150 \
    --max_length=2048 \
    --max_prompt_length=2000 \
    --num_train_epochs=1 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./summarization.py \
    --output_file="${FOLDER}/${TARGET_MODEL}_${SPLIT}_${DECODING_TYPE}" \
    --decoding_type="${DECODING_TYPE}" \
    --model="models_gpt/${TARGET_MODEL}/final_checkpoint" \
    --tokenizer_model="${TOKENIZER_MODEL}" \
    --batch_size=1 \
    --dataset="${DATASET}" \
    --split="${SPLIT}" \

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./evaluation.py \
    --input_file "${FOLDER}/${TARGET_MODEL}_${SPLIT}_${DECODING_TYPE}.json" \
    --output_file "${FOLDER}/evaluation.json" \
    --batch_size 16 \
    --alignscore_ckpt "${PATH_TO_ALIGNSCORE_CKPT}" \
