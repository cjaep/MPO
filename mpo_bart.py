# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed, AutoModelForSeq2SeqLM

from trl import DPOTrainer
from collections import defaultdict
import json


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    rejected_file: Optional[str] = field(default='./results/llama_greedy.json', metadata={"help": "rejected file"})
    chosen_file: Optional[str] = field(default='./results/llama_beam.json', metadata={"help": "chosen file"})
    # data parameters
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for DPO loss"})

    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "total train epochs"})
    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=150, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )


def get_xsum_paired(
    rejected_file,
    chosen_file,    
    train_ratio=0.9
    )->Dataset:

    train_data_dict = defaultdict(list)
    eval_data_dict = dict()

    with open(rejected_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['source'].strip()=="": continue
            train_data_dict['prompt'].append(data['source'])
            train_data_dict['rejected'].append(data['predicted'].strip())
    with open(chosen_file, 'r') as f:
        for line in f:
            data = json.loads(line)       
            if data['source'].strip()=="": continue
            train_data_dict['chosen'].append(data['predicted'].strip())

    # split
    thres = int(len(train_data_dict['prompt']) * train_ratio)
    for k in train_data_dict.keys():
        train_data_dict[k], eval_data_dict[k] = train_data_dict[k][:thres], train_data_dict[k][thres:]

    train_dataset = Dataset.from_dict(dict(train_data_dict))
    eval_dataset = Dataset.from_dict(eval_data_dict)

    return train_dataset, eval_dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    print(f'script_args:\n{script_args}\n')

    set_seed(script_args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForSeq2SeqLM.from_pretrained(
        script_args.model_name_or_path,
        device_map={"": Accelerator().local_process_index},
        output_attentions=True
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum", forced_bos_token_id=0)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = get_xsum_paired(script_args.rejected_file, script_args.chosen_file)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs = script_args.num_train_epochs, 
        # max_steps=len(train_dataset)//script_args.per_device_train_batch_size,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_bart_xsum_sft",
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
    )

    peft_config = None

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)