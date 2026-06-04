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
            gemini_model="gemini-2.5-flash"
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

        Where relevant, reference these entities directly in the report only if they appear in the insights with supporting context. If an entity appears in the knowledge graph but has no supporting detail in the insights, do not include it.

        Insights:
        {insights}

        Format:
        Title
        Introduction
        Key Trends
        Industry Leaders
        Future Outlook
        Conclusion
        Your report must follow this exact structure:
        ## Summary
        (2-3 sentence overview)

        ## Key Trends
        (bullet points)

        ## Key Players
        Only include companies or organisations that are explicitly mentioned in the insights above with specific context about what they do or have done. If no companies are mentioned with specific detail, write "No specific organisations identified in this research." Do not list company names without context and do not use placeholder text.

        ## Statistics
        (any numbers or data points found)

        ## Conclusion
        (1-2 sentences)"""

        if self._finetuned:
            return self._finetuned.generate(prompt)

        return self.run(prompt)
