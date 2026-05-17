from .base_agent import BaseAgent


class AnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Data Analyst",
            goal="Extract meaningful insights from research"
        )

    def analyze(self, documents, query):
        # pull anything relevant from previous searches
        try:
            past_docs = self.mcp.call_tool("vector_store", "search", {"query": query, "top_k": 5}) or []
        except Exception as e:
            print(f"[analyst] vector store unavailable: {e}")
            past_docs = []

        current = ""
        for doc in documents[:5]:
            current += f"Title: {doc['title']}\nContent: {doc['content']}\n\n"

        past = "\n".join(
            doc["content"] if isinstance(doc, dict)
            else doc
            for doc in past_docs
        )

        prompt = f"""Use the research below to extract insights.
        Think through this step by step:
        1. First identify the main themes across all sources
        2. Then look for supporting statistics or data points
        3. Then identify key organisations or companies involved
        4. Finally summarise the key trends

        Current research:
        {current}

        Relevant context retrieved from memory (ranked by relevance):
        {past}

        Pull out:
        - key trends
        - any statistics worth noting
        - major companies or organisations mentioned
        """
        return self.run(prompt)
