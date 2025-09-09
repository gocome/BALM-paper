from datetime import date
import os
import torch
import warnings
warnings.simplefilter('ignore')

from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from datasets import load_dataset


run_name = f"BALM-paired_lc-coherence-data_90-5-5-split_{date.today().isoformat()}"
print(f"Run name: {run_name}")

use_fp16 = torch.cuda.is_available()  # Only enable if CUDA is available
print(f"Using fp16: {use_fp16}")

balm_config = {
    "run_name": run_name,
    
    # model architecture
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "vocab_size": 25,
    "max_len": 512,
    "max_position_embeddings": 514,
    
    # tokenizer
    "padding": "max_length",
    "truncate": True,
    "return_special_tokens_mask": True,
    
    # training parameters
    "batch_size": 32,
    "max_steps": 50,
    "warmup_steps": 3,
    "weight_decay": 0.01,
    "peak_learning_rate": 4e-4,
    "adam_epsilon": 1e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "type_vocab_size": 2,  # this should be 2 for paired/mixed models, 1 for unpaired models
    "fp16": use_fp16,
    "evaluation_strategy": "steps",
    "seed": 42,
    
    # outputs and logging
    "save_steps": 10,
    "eval_steps": 5,
    "output_dir": f"./checkpoints/{run_name}",  # where the checkpoint data will be written
    "logging_dir": f"./logs/{run_name}",
    "logging_steps": 100,
    "overwrite_output_dir": True,
    "logging_first_step": True,
}

# initialize the model using the BALM config dictionary
# defaults are based on what was used in the paper
model_config = RobertaConfig(
    vocab_size=balm_config.get("vocab_size", 25),
    hidden_size=balm_config.get("hidden_size", 1024),
    intermediate_size=balm_config.get("intermediate_size", 4096),
    max_position_embeddings=balm_config.get("max_position_embeddings", 512),
    num_hidden_layers=balm_config.get("num_hidden_layers", 24),
    num_attention_heads=balm_config.get("num_attention_heads", 16),
    type_vocab_size=balm_config.get("type_vocab_size", 2),
)
    
model = RobertaForMaskedLM(model_config)

model_size = sum(p.numel() for p in model.parameters())
print(f"Model size: {model_size/1e6:.2f}M")

# load the tran, eval, and test data
data_files = {
    "train": ['./data/train-test-eval_paired/train.txt'],
    "eval": ['./data/train-test-eval_paired/eval.txt'],
    "test": ['./data/train-test-eval_paired/test.txt']
}

dataset = load_dataset("text", data_files=data_files)

tokenizer = RobertaTokenizer.from_pretrained("tokenizer")

tokenized_dataset = dataset.map(
    lambda x: tokenizer(
        x["text"],
        padding=balm_config.get("padding", "max_length"),
        truncation=balm_config.get("truncation", True),
        max_length=balm_config.get("max_len", 512),
        return_special_tokens_mask=balm_config.get("return_special_tokens_mask", True),
    ),
    remove_columns=["text"],
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    fp16=balm_config.get("fp16", True),
    eval_strategy=balm_config.get("evaluation_strategy", "steps"),
    seed=balm_config.get("seed", 42),
    per_device_train_batch_size=balm_config.get("batch_size", 32),
    per_device_eval_batch_size=balm_config.get("batch_size", 32),
    max_steps=balm_config.get("max_steps", 500000),
    save_steps=balm_config.get("save_steps", 100000),
    logging_steps=balm_config.get("logging_steps", 100),
    eval_steps=balm_config.get("eval_steps", 25000),
    adam_beta1=balm_config.get("adam_beta1", 0.9),
    adam_beta2=balm_config.get("adam_beta2", 0.98),
    adam_epsilon=balm_config.get("adam_epsilon", 1e-6),
    weight_decay=balm_config.get("weight_decay", 0.01),
    warmup_steps=balm_config.get("warmup_steps", 30000),
    learning_rate=balm_config.get("peak_learning_rate", 4e-4),
    gradient_accumulation_steps=balm_config.get("gradient_accumulation_steps", 1),
    
    # output and logging
    run_name=balm_config.get("run_name", None),
    output_dir=balm_config.get("output_dir", f"./checkpoints/{run_name}"),
    overwrite_output_dir=balm_config.get("overwrite_output_dir", True),
    logging_dir=balm_config.get("logging_dir", f"./logs/{run_name}"),
    report_to=balm_config.get("report_to", None),
    logging_first_step=balm_config.get("logging_first_step", True),
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"]
)

trainer.train()

trainer.save_model(f"../models/{run_name}")
