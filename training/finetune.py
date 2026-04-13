"""
Fine-tunes phi-2 using LoRA on the generated dataset.

Usage:
    python training/finetune.py

Requirements:
    pip install transformers datasets peft accelerate

Notes:
    - CPU training works but is slow (~30-60 min for this dataset size)
    - If you have a GPU, it will be used automatically
    - The adapter is saved to models/lora-adapter/
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

#  config 

BASE_MODEL = "facebook/opt-125m"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "lora-adapter")
MAX_LENGTH = 512

# load dataset 

print("loading dataset...")
with open(DATASET_PATH) as f:
    raw = json.load(f)

# format each example as a single training string
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

formatted = [format_example(ex) for ex in raw]
dataset = Dataset.from_list(formatted)
print(f"loaded {len(dataset)} examples")

#  load model + tokenizer 

print(f"loading {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,  # use float16 if you have a GPU
    trust_remote_code=True,
)

#  apply lora 

print("applying LoRA...")
lora_config = LoraConfig(
    r=8,                          # rank — higher = more capacity, more memory
    lora_alpha=16,                # scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#  tokenize 

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("tokenizing...")
tokenized = dataset.map(tokenize, remove_columns=["text"])

#  train 

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # simulates batch size of 4 on limited memory
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    fp16=False,                     # set True if you have a GPU
    report_to="none",               # disable wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("starting training...")
trainer.train()

#  save 

print(f"saving adapter to {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("done")
