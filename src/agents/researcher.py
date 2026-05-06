import time
from .base_agent import BaseAgent
import traceback

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Research Specialist",
            goal="Collect concise information from online sources"
        )
        

    def search(self, query, max_results=3, retries=3, delay=2):
        for attempt in range(retries):
            try:
                documents = self.mcp.call_tool("web_search", "search", {
                    "query": query,
                    "max_results": max_results,
                    "retries": retries,
                    "delay": delay
                })
                if documents:
                    self.mcp.call_tool("vector_store", "add", {"documents": documents})
                    return documents
            except Exception as e:
                print(f"[researcher] search attempt {attempt + 1} failed: {e}")
                traceback.print_exc()
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
