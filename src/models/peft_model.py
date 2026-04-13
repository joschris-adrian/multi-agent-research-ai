"""
Loads the fine-tuned LoRA adapter on top of the base phi-2 model.
Falls back gracefully if the adapter hasn't been trained yet.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "facebook/opt-125m"
ADAPTER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "lora-adapter"
)


class FineTunedWriter:
    def __init__(self):
        adapter_path = os.path.abspath(ADAPTER_PATH)

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_path}. "
                "Run `python training/finetune.py` first."
            )

        print(f"loading base model: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
            trust_remote_code=True,
        )

        print(f"loading LoRA adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        print("fine-tuned writer ready")

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # strip the prompt from the output
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
