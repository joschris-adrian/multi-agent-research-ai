"""
Compares report quality before and after fine-tuning.

Usage:
    python training/evaluate_finetuning.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.base_agent import BaseAgent
from src.evaluation.evaluator import Evaluator

TEST_TOPICS = [
    "What are the latest trends in solar energy?",
    "How is AI being used in climate research?",
]

evaluator = Evaluator()
base_writer = BaseAgent(
    role="Technical Writer",
    goal="Write a well-structured research report"
)

def write_with_ollama(topic):
    return base_writer.run(
        f"Write a structured research report on: {topic}\n"
        "Format: Title, Introduction, Key Trends, Industry Leaders, Future Outlook, Conclusion"
    )

def write_with_lora(topic):
    from src.models.peft_model import FineTunedWriter
    writer = FineTunedWriter()
    return writer.generate(f"Write a structured research report on: {topic}")

results = []

print("=" * 60)
print("FINE-TUNING EVALUATION")
print("Baseline: Ollama (llama3.2)")
print("Fine-tuned: opt-125m + LoRA (10 examples, 3 epochs)")
print("=" * 60)

for topic in TEST_TOPICS:
    print(f"\nTopic: {topic}\n")

    print("--- Ollama (baseline) ---")
    ollama_output = write_with_ollama(topic)
    ollama_score = evaluator.evaluate(topic, ollama_output)
    print(ollama_output[:400] + "...\n")
    print(f"Score:\n{ollama_score}")

    print("\n--- Fine-tuned LoRA (opt-125m) ---")
    try:
        lora_output = write_with_lora(topic)
        lora_score = evaluator.evaluate(topic, lora_output)
        print(lora_output[:400] + "...\n")
        print(f"Score:\n{lora_score}")
        results.append({
            "topic": topic,
            "ollama_output": ollama_output,
            "ollama_score": ollama_score,
            "lora_output": lora_output,
            "lora_score": lora_score,
        })
    except FileNotFoundError as e:
        print(f"skipping: {e}")

    print("\n" + "=" * 60)

print("""
NOTES:
- opt-125m is a 125M parameter model fine-tuned on 10 examples
- The baseline uses llama3.2 (3B parameters) via Ollama
- The purpose of this evaluation is to demonstrate the full
  fine-tuning pipeline, not to beat a much larger model
- A fair comparison would require fine-tuning a model of
  similar size to llama3.2, which needs significantly more
  compute and training data
""")
