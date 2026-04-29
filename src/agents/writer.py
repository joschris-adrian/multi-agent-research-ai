import os
from .base_agent import BaseAgent

USE_FINETUNED = os.getenv("USE_FINETUNED", "0") == "1"


class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Technical Writer",
            goal="Write a well-structured research report",
            temperature=0.8,
            max_tokens=800,
        )
        self._finetuned = None

        if USE_FINETUNED:
            try:
                from src.models.peft_model import FineTunedWriter
                self._finetuned = FineTunedWriter()
                print("[writer] using fine-tuned LoRA model")
            except FileNotFoundError as e:
                print(f"[writer] {e}")
                print("[writer] falling back to Ollama")

    def write_report(self, insights: str, entities: dict = None) -> str:
        entities = entities or {}

        companies = ", ".join(entities.get("companies", [])) or "not identified"
        trends = ", ".join(entities.get("trends", [])) or "not identified"
        technologies = ", ".join(entities.get("technologies", [])) or "not identified"

        prompt = f"""Write a structured research report using the insights and entities below.

Key entities extracted from the knowledge graph:
- Companies: {companies}
- Trends: {trends}
- Technologies: {technologies}

Where relevant, reference these entities directly in the report rather than inventing new ones.

Insights:
{insights}

Format:
Title
Introduction
Key Trends
Industry Leaders
Future Outlook
Conclusion"""

        if self._finetuned:
            return self._finetuned.generate(prompt)

        return self.run(prompt)
