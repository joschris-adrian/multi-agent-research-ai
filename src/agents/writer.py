import os
from .base_agent import BaseAgent

USE_FINETUNED = os.getenv("USE_FINETUNED", "0") == "1"


class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Technical Writer",
            goal="Write a well-structured research report"
        )
        self._finetuned = None

        if USE_FINETUNED:
            try:
                # import here so torch is only loaded when actually needed
                from src.models.peft_model import FineTunedWriter
                self._finetuned = FineTunedWriter()
                print("[writer] using fine-tuned LoRA model")
            except FileNotFoundError as e:
                print(f"[writer] {e}")
                print("[writer] falling back to Ollama")

    def write_report(self, insights: str) -> str:
        prompt = f"""Write a structured research report using the following insights.

Format:
Title
Introduction
Key Trends
Industry Leaders
Future Outlook
Conclusion

Insights:
{insights}"""

        if self._finetuned:
            return self._finetuned.generate(prompt)

        return self.run(prompt)
