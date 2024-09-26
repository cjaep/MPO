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
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
)
import argparse
import json
from datasets import load_dataset
from torch.utils.data import DataLoader



def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='results/output_greedy.json')
    parser.add_argument('--model', type=str, default='/home/jiwoosong/trl/dpo_bart_default/final_checkpoint')
    parser.add_argument('--decoding_type', type=str, default='greedy')
    parser.add_argument('--num_beams', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--split', type=str, default='test')

    return parser

def batch_writer(output_file, doc_id, summary, gold, source):
    for i in range(len(doc_id)):
        output_dict_example = {
            "id" : doc_id[i],
            "predicted" : summary[i],
            "gold" : gold[i],
            "source" : source[i],
        }
        with open(f"{output_file}.json", "a") as _jsonl_file:
            _jsonl_file.write(json.dumps(output_dict_example))
            _jsonl_file.write("\n")

def main(args, device):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    if args.model == 'sft_bart_xsum':
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum", output_attentions=True)
        model.to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model,
            device_map=device,
        )
    model.eval()

    dataset = load_dataset("EdinburghNLP/xsum", split=args.split)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    for input_item in tqdm(test_dataloader):

        input_text = input_item["document"] 
        # for j in range(len(input_text)):
        #     input_text[j] = "Document: " + input_text[j]
        doc_id = input_item["id"]           
        gold = input_item["summary"]        

        model_inputs = tokenizer(input_text, max_length=args.model_max_length, padding=True, truncation=True, return_tensors='pt').to(device)
        # 1.0, 3.0, 5.0
        if args.decoding_type=='greedy':
            output_token = model.generate(**model_inputs, max_new_tokens=50)

        elif args.decoding_type=='beam':
            output_token = model.generate(
                **model_inputs, 
                num_beams=args.num_beams,
                early_stopping=True,
                max_new_tokens=50,
            )

        elif args.decoding_type=='temp5':
            output_token = model.generate(
                **model_inputs, 
                max_new_tokens=50, 
                do_sample=True,
                temperature=5.0,
            )

        elif args.decoding_type=='temp3':
            output_token = model.generate(
                **model_inputs, 
                max_new_tokens=50, 
                do_sample=True,
                temperature=3.0,
            )

        elif args.decoding_type=='temp1':
            output_token = model.generate(
                **model_inputs, 
                max_new_tokens=50, 
                do_sample=True,
                temperature=1.0,
            )

        documents = tokenizer.batch_decode(output_token, skip_special_tokens=True)

        output_dict_example = {
            "output_file" : args.output_file,
            "doc_id" : doc_id,
            "summary" : documents,            
            "gold" : gold,
            "source" : input_text,
        }

        batch_writer(**output_dict_example)

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    print(f'args: \n{args}\n')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    main(args, device)