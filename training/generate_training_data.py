"""
Run this first to generate training data using your existing agents.
This creates domain-specific examples for fine-tuning the writer.

Usage:
    python training/generate_training_data.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.base_agent import BaseAgent

TOPICS = [
    "renewable energy trends",
    "electric vehicles and battery technology",
    "solar panel efficiency improvements",
    "wind energy expansion",
    "hydrogen fuel cells",
    "energy storage systems",
    "smart grid technology",
    "carbon capture methods",
    "nuclear fusion research",
    "offshore wind farms",
]

generator = BaseAgent(
    role="Research Writer",
    goal="Write structured research reports"
)

dataset = []

for topic in TOPICS:
    print(f"generating example for: {topic}")

    prompt = f"""Write a structured research report on: {topic}

Use this exact format:

Title: [title here]

Introduction:
[2-3 sentences introducing the topic]

Key Trends:
[3-4 bullet points on current trends]

Industry Leaders:
[2-3 key companies or organisations]

Future Outlook:
[2-3 sentences on what comes next]

Conclusion:
[1-2 sentences wrapping up]
"""

    response = generator.run(prompt)

    dataset.append({
        "instruction": f"Write a structured research report on: {topic}",
        "output": response
    })

    print(f"done: {topic}\n")

output_path = os.path.join(os.path.dirname(__file__), "dataset.json")
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"saved {len(dataset)} examples to {output_path}")
