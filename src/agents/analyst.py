from .base_agent import BaseAgent
from ..memory.vector_store import VectorStore


class AnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Data Analyst",
            goal="Extract meaningful insights from research"
        )
        self.memory = VectorStore()

    def analyze(self, documents, query):
        # pull anything relevant from previous searches
        past_docs = self.memory.search(query)

        current = ""
        for doc in documents[:5]:
            current += f"Title: {doc['title']}\nContent: {doc['content']}\n\n"

        past = "\n".join(past_docs)

        prompt = f"""Use the research below to extract insights.

Current research:
{current}

Previous searches on related topics:
{past}

Pull out:
- key trends
- any statistics worth noting
- major companies or organisations mentioned
"""
        return self.run(prompt)
