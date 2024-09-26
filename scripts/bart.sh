#!/bin/bash

GPU="0"
FOLDER="results_bart"
CHOSEN_FILE="./results_bart/xsum_bart_SFT_validation_beam.json"
REJECTED_FILE="./results_bart/xsum_bart_SFT_validation_temp5.json"
BASE_MODEL="facebook/bart-large-xsum"
TARGET_MODEL="xsum_bart_MPO"
DECODING_TYPE="beam"
SPLIT="test"
DATASET="EdinburghNLP/xsum"
PATH_TO_ALIGNSCORE_CKPT=""

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./mpo_bart.py \
    --rejected_file="${REJECTED_FILE}" \
    --chosen_file="${CHOSEN_FILE}" \
    --model_name_or_path="${BASE_MODEL}" \
    --output_dir="models_bart/${TARGET_MODEL}" \
    --beta=0.5 \
    --learning_rate=1e-6 \
    --warmup_steps=150 \
    --max_length=512 \
    --max_prompt_length=1024 \
    --num_train_epochs=3 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./summarization_bart.py \
    --output_file="${FOLDER}/${TARGET_MODEL}_${SPLIT}_${DECODING_TYPE}" \
    --decoding_type="${DECODING_TYPE}" \
    --model="models_bart/${TARGET_MODEL}/final_checkpoint" \
    --split="${SPLIT}" \

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./evaluation.py \
    --input_file "${FOLDER}/${TARGET_MODEL}_${SPLIT}_${DECODING_TYPE}.json" \
    --output_file "${FOLDER}/evaluation.json" \
    --batch_size 16 \
    --alignscore_ckpt "${PATH_TO_ALIGNSCORE_CKPT}" \
