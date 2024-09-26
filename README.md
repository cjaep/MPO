# Model-based Preference Optimization (MPO)
Official code for the EMNLP 2024 paper **Model-based Preference Optimization in Abstractive Summarization without Human Feedback**

<p align="center">
<img alt="teaser" img src="https://lh3.googleusercontent.com/fife/ALs6j_Hy5-HLZln_CL4t22m72hqbQ14DGqWU_dDF3rUHPsdsIOOydZQH5pFUPMsD28RerHxWyP3fPmceftDymHV7pwV-MbylfbIx07t8bIeCU02hIUIxjpan_ZOygVRU_X83SFwYDJCCkooNmUiyaQEPemnzdNvyuluNEBhq_bxQMbLcohukMfjyZXGtOaCLzK_5vkLCs9LFU5f7dcCArCHgSCNHL7TjvF0pskDGWDlvvghcPKYLT4k_uhJtB70I7VXZirwOCd4oypXXXUip0i9WLrMoULbYyosu3QrfpZcnQ3lfci4d-eAiQH9rsxwKXJxluuKpUAJ7LbdMSXAAjl2jwHb_WS0VzqUKot-gD5v1UuC35aXjc1Dh8QTXb-WE39cPh6lo6qNifTbIOD7Qgzxw-vuGVRV50gwl_hbq_eH4zKHYMWl3mwTPazMDI4z7PF_AA3X5TF0cm_zdQRNWhJeT-agY8NzvC0ZC_cY_f3xKqFkz9DI3ruF4rcIqLnXuhUKHYeTTWanFU8w7U7loMa_qSRp2I1UimKORb-g-GYJuVjwF9ejatruSb6svQlvfQjLAxwZIVotSyzfpQWwk75U5-OEbB0wEF0owgHiUw0_l4oE5lwyijBPksqof2tUMveH8jE_3_ZCGfFsbYR8kDUfRunTfs_u7XE3ZS8wS02AK3qBQRV0q2cU73zc7QLtSQABOJAO3Y8gkSebK57c7kxNTq9lveeEKlHWgxjHodYgWNcJLzyJbndW6_j6E-HXRdjbZzodBgoE27TmwTn103wiXGRxKgORCYyXPhTaEnTpM4rTLU58deSG38DYDghZyVRhF80GL2X1MxjMtLTYCDnyAwFtM7iTZPU1dE6rUYkiiSIXVovT6l9ck2nm0t8z4_kGdq5VbTOMqOuuBMW95OKt1rlb917BPwRPE_4jnzw0ShmYTNJ1Uq4Wu9rxgljD8Ho-mnObvI-s0W5Nb2MUAUU4MNMnqGE4YJLKI_orv7QyO8yhDs-4vz0EYk7PGTl1CidpPmg3U04lsUMiY2YGlwRWWy8pHVQ7jan9dbohlouHe6VEPVaOv5ua2-Tusx6T-joNBvSLMzQh7KxlYiIuEwJqRsfr4-r7LVurS3wnLJrimz6KbPVyVpaG7ZXnbfeLGAfWd8mBiAmLQaqvSDp2-nZMuhmUzSuwjMai8eR53RNOG8fHFPkfEOwKZq3n-rs-tst-aC4qMrsC4_UaQ49cAFQRGimgB-vo5cXCUK5uLNv0N8JrMjTxTfva4EEgWR7WhSIvZ3gsVSWKt0UhHoKrFTr6-j6dFCOrlDUYbXP5AQ5Rz7roepEwMwqMqFWg1a2fOhW0-GNRdAmKVN_yaM1ajHiyUaZ0skcBrkWTNV_u66L-RYnOjknRA1TuPIPg2thlefW1TAnQTNpOTKiiw3bpZQw-zXJUV0V-WvMm2w-3_G3Penv42OURFIRNnNHGWn2XIideFXY5WGimKzxXdElabOzFYb5vn7yokVHyflQoTg7T-OlTfmxF_wx8wc_QLucdGIYwYogTz48_LULNDKMSeRJm16WQAL0G53HPAjbVvAenVH9y4XBB_Q3z30jTN8xkhuVK34IrXkaUyiVVcGHjKClQGSqzZE0ElVuSOexLIHpdkKtRyCRQ1S6Q1ocKBcaXMqffMPMF0Gn4H8opdo556om80Wz-RP88=w1917-h1904" width="100%">
</p>

* We introduced a novel approach, **Model-based Preference Optimization (MPO)** to fine-tune LLMs for improved summarization abilities **without any human feedback**.
* By leveraging the model's inherent summarization capabilities, we created a preference dataset that is fully generated by models using **different decoding strategies**.
* Our proposed MPO significantly enhances the quality of generated summaries without relying on human feedback.

## Installation 
Our code is based on Huggingface's `transformers>=4.35.0`.

```bash
conda create -n MPO python=3.8
conda activate MPO
git clone https://github.com/cjaep/MPO.git
cd MPO
pip install -r requirements.txt
```

## Quick Start
This script will perform MPO on the TL;DR dataset using the GPT-J model.
```sh
mkdir models_gpt
bash scripts/gpt.sh
```

This script will perform MPO on the XSUM dataset using the BART model.
```sh
mkdir models_bart
bash scripts/bart.sh
```

## Step-by-step Instructions
To perform MPO, follow these three steps:
1. **SFT**: Fine-tune the LLMㄴ on an abstractive summarization task to generate a single-sentence summary from the given source text.
2. **DPO**: Using the SFT model trained in the previous stage, generate single-sentence summaries from the **validation split** of the dataset through **beam search** and **temperature-scaled sampling**. Summaries generated with beam search are used as chosen samples, while those generated with temperature-scaled sampling are used as rejected samples. These data pairs are then used to fine-tune the model through DPO.
3. **Evaluation**: For the model trained in the DPO stage, generate summaries using the test split of the dataset and evaluate their quality using five different automatic metrics.

### 1. SFT
The implementation follows an example from https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts

```sh
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python ./sft.py \
    --model_name="mistralai/Mistral-7B-v0.1" \
    --output_dir="./models/sft_mistral_tldr" \
    --dataset_name="CarperAI/openai_summarize_tldr" \
    --max_steps=500 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="sft" \
    --report_to="wandb" \
```

### 2. DPO
```sh
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python ./mpo.py \
    --rejected_file="${REJECTED_FILE}" \
    --chosen_file="${CHOSEN_FILE}" \
    --model_name_or_path="models/${BASE_MODEL}" \
    --output_dir="models/${TARGET_MODEL}" \
    --beta=0.5 \
    --learning_rate=1e-4 \
    --warmup_steps=150 \
    --max_length=2048 \
    --max_prompt_length=2000 \
    --num_train_epochs=1 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
```

### 3. Summarization
```sh
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python ./summarization.py \
    --output_file="${FOLDER}/${TARGET_MODEL}_${SPLIT}_${DECODING_TYPE}" \
    --decoding_type="${DECODING_TYPE}" \
    --model="models/${TARGET_MODEL}/final_checkpoint" \
    --tokenizer_model="${TOKENIZER_MODEL}" \
    --batch_size=1 \
    --dataset="${DATASET}" \
    --split="${SPLIT}" \
```

### 4. Evaluation

To run the evaluation, install following metrics:
- [AlignScore](https://github.com/yuh-zha/AlignScore)
- [BARTScore](https://github.com/neulab/BARTScore)


```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python ./evaluation.py \
    --input_file "${FOLDER}/${TARGET_MODEL}_${SPLIT}_${DECODING_TYPE}.json" \
    --output_file "${FOLDER}/evaluation.json" \
    --batch_size 16 \
    --alignscore_ckpt "${PATH_TO_ALIGNSCORE_CKPT}" \
```
