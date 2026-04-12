import time
from .base_agent import BaseAgent
from ddgs import DDGS
from ..memory.vector_store import VectorStore


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Research Specialist",
            goal="Collect concise information from online sources"
        )
        self.memory = VectorStore()

    def search(self, query, max_results=3, retries=3, delay=2):
        for attempt in range(retries):
            try:
                documents = []
                with DDGS() as ddgs:
                    results = ddgs.text(query, max_results=max_results)
                    for r in results:
                        documents.append({
                            "title": r["title"],
                            "content": r["body"][:500],
                            "source": r["href"]
                        })

                if documents:
                    self.memory.add_documents(documents)
                    return documents

            except Exception as e:
                print(f"[researcher] search attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)

        print("[researcher] all search attempts failed, returning empty")
        return []

    def extract_query(self, tasks: str, original_question: str) -> str:
        prompt = f"""Given this research question: "{original_question}"
And these research tasks: {tasks}

Return ONLY a short 5-10 word search query suitable for a web search engine.
No explanation, no punctuation, just the query itself."""
        return self.run(prompt).strip()
