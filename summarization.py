import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    BeamSearchScorer,
    set_seed,
)
import argparse
import json
from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import datetime


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--decoding_type', type=str, default='greedy')
    parser.add_argument('--num_beams', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default="CarperAI/openai_summarize_tldr_sft")
    parser.add_argument('--tokenizer_model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--dataset', type=str, default="CarperAI/openai_summarize_tldr")
    parser.add_argument('--split', type=str, default='test')

    return parser

def batch_writer(output_file, summary, gold, source):
    for i in range(len(gold)):
        output_dict_example = {
            "predicted" : summary[i],
            "gold" : gold[i],
            "source" : source[i],
        }
        with open(f"{output_file}.json", "a") as _jsonl_file:
            _jsonl_file.write(json.dumps(output_dict_example))
            _jsonl_file.write("\n")
    return

def main(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"     

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    if args.model is None or args.model=="sft":
        # print("CarperAI/openai_summarize_tldr_sft")
        model = AutoModelForCausalLM.from_pretrained("CarperAI/openai_summarize_tldr_sft", local_files_only=True, device_map="auto", quantization_config=bnb_config)
    elif args.model=="ppo":
        # print("CarperAI/openai_summarize_tldr_ppo")
        model = AutoModelForCausalLM.from_pretrained("CarperAI/openai_summarize_tldr_ppo", device_map="auto", quantization_config=bnb_config)
    elif args.model=="plm":
        model = AutoModelForCausalLM.from_pretrained(args.tokenizer_model, device_map="auto", quantization_config=bnb_config)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
        ).to("cuda")
        if "EdinburghNLP/xsum" in args.dataset and "sft" not in args.model:
            model = model.merge_and_unload() 

    dataset = load_dataset(args.dataset, split=args.split)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for i, input_item in enumerate(tqdm(test_dataloader, desc="Predicting...")): 
            if "CarperAI/openai_summarize_tldr" in args.dataset:
                if i==1: print("TLDR dataset")
                input_text = input_item["prompt"]
                gold = input_item["label"]
                model_inputs = tokenizer(input_text, max_length=2000, padding=True, truncation=True, return_tensors='pt').to("cuda")
                
            elif "EdinburghNLP/xsum" in args.dataset:
                if i==1: print("XSUM dataset")
                input_text = input_item["document"]
                gold = input_item["summary"] 
                model_inputs = tokenizer(input_text, max_length=1950, return_tensors='pt')
                temp = tokenizer.batch_decode(model_inputs['input_ids'])
                if "gpt" in args.tokenizer_model:
                    if i==1: print("EleutherAI/gpt-j-6b tokenizer")
                    for j in range(len(temp)):
                        # temp[j] = 'Document: ' + temp[j] + '\n\nSummary: '
                        temp[j] = 'Document: ' + temp[j] + '\n\nSummarize the article in one sentence. Summary: '
                else:
                    if i==1: print("llama or mistral tokenizer")
                    for j in range(len(temp)):
                        # temp[j] = 'Document: ' + temp[j][4:] + '\n\nSummary: '  # Avoid overlapping <SOS> token
                        temp[j] = 'Document: ' + temp[j][4:] + '\n\nSummarize the article in one sentence. Summary: '  # Avoid overlapping <SOS> token
                model_inputs = tokenizer(temp, truncation=True, return_tensors='pt').to("cuda")
            
            if args.decoding_type=='greedy':
                output_token = model.generate(**model_inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

            elif args.decoding_type=='temp5':
                output_token = model.generate(
                    **model_inputs, 
                    max_new_tokens=50, 
                    do_sample=True,
                    temperature=5.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            elif args.decoding_type=='temp3':
                output_token = model.generate(
                    **model_inputs, 
                    max_new_tokens=50, 
                    do_sample=True,
                    temperature=3.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            elif args.decoding_type=='temp1':
                output_token = model.generate(
                    **model_inputs, 
                    max_new_tokens=50, 
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            elif args.decoding_type=='beam':
                output_token = model.generate(
                    **model_inputs, 
                    num_beams=args.num_beams,
                    early_stopping=True,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id,
                )

            elif args.decoding_type=='nucleus':
                output_token = model.generate(
                    **model_inputs, 
                    max_new_tokens=50, 
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            documents = tokenizer.batch_decode(output_token, skip_special_tokens=True)
                    
            if "CarperAI/openai_summarize_tldr" in args.dataset:
                for k in range(len(documents)):
                    documents[k] = documents[k][len(input_text[k]):].strip()

            elif "EdinburghNLP/xsum" in args.dataset:
                for k in range(len(documents)):
                    documents[k] = documents[k][len(temp[k]):].strip()

            output_dict_example = {
                "output_file" : args.output_file,
                "summary" : documents,            
                "gold" : gold,
                "source" : input_text,
            }
            batch_writer(**output_dict_example)

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    main(args)