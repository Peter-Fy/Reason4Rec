import pandas as pd
import json 
import re
from tqdm import tqdm
import numpy as np
from unsloth import FastLanguageModel
import os, wandb
from trl.commands.cli_utils import TrlParser
import torch
from datasets import Dataset
from transformers import set_seed, TrainingArguments, DataCollatorForSeq2Seq
from trl import (
    SFTConfig, 
    SFTTrainer
)
from datasets import load_dataset
from dataclasses import field, dataclass
import pickle
from unsloth.chat_templates import train_on_responses_only


@dataclass
class FinetuneArguments:
    model_path: str = field()
    train_data_path: str = field()
    train_size: int = field()
    test_size: int = field()
    max_len: int = field()
    lora_rank: int = field()
    lora_alpha: int = field()
    wandb_project: str = field()
    load_in_4bit: bool = field()
    target_modules: str = field()
    tune_last_layer: bool = field()
    mask_loss: bool = field()
    lora_dropout: float = field()

def main():
    parser =  TrlParser((FinetuneArguments, SFTConfig))
    args, training_args = parser.parse_args_and_config()
    args.target_modules = args.target_modules.split(",")
    if training_args.report_to:
        if training_args.report_to[0] == 'wandb':
            print("Logging to wandb")
            wandb.login(key="your_key_here")
            wandb.init(project=args.wandb_project, job_type = "training")
    print("=====================================================================================================")
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    for arg in vars(training_args):
        if arg in ["lora_rank", "lora_alpha", "learning_rate", "weight_decay", "eval_steps", "save_steps"]:
            print(f"{arg}: {getattr(training_args, arg)}")
    print("=====================================================================================================")
    
    # Load Model && Tokenizer
    max_seq_length = args.max_len
    set_seed(training_args.seed)
    np.random.seed(training_args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = args.target_modules, 
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout, 
        bias = "none",   
        use_gradient_checkpointing = "unsloth", 
        random_state = training_args.seed,
        use_rslora = False,  
        loftq_config = None, 
    )

    if tokenizer.pad_token_id is None:      
        print(f"set pad_token_id to 0")
        tokenizer.pad_token_id = 0
    
    # Load Data
    train_dataset = pickle.load(open(args.train_data_path, 'rb'))
    train_dataset = Dataset.from_list(train_dataset)

    if args.train_size > 0:
        train_dataset = train_dataset.select(range(args.train_size))

    if args.tune_last_layer:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    if args.mask_loss:
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            max_seq_length = args.max_len,
            dataset_num_proc = 2,
            packing = False, 
            args = training_args,
            data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    else:
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            max_seq_length = args.max_len,
            dataset_num_proc = 2,
            packing = False, 
            args = training_args,
        )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()