import json
from .base_agent import BaseAgent


class GraphBuilderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Knowledge Graph Builder",
            goal="Extract structured entities and relationships from research insights",
            temperature=0.1,  # low temperature for consistent JSON output
            max_tokens=600,
        )

    def extract_entities(self, insights: str, topic: str) -> dict:
        prompt = f"""Read the following research insights about "{topic}" and extract entities.

Return ONLY valid JSON in this exact format, nothing else:
{{
  "companies": ["Company A", "Company B"],
  "trends": ["Trend A", "Trend B"],
  "technologies": ["Tech A", "Tech B"],
  "relationships": [
    {{"source": "Company A", "target": "Tech A", "relation": "USES"}},
    {{"source": "Trend A", "target": "Tech B", "relation": "DRIVES"}}
  ]
}}

Insights:
{insights}
"""
        raw = self.run(prompt)

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            print("[graph_builder] could not parse entity JSON, skipping")
            return {"companies": [], "trends": [], "technologies": [], "relationships": []}
