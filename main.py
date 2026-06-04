import sys
import os
from dotenv import load_dotenv
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.workflow.agent_pipeline import MultiAgentResearchSystem

def main():
    load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    print(f"LLM provider: {provider}")
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY", ""):
        print("Error: GEMINI_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)
    question = input("Enter a research question: ")
    system = MultiAgentResearchSystem()
    result = system.run(question)
    print("\n===== FINAL REPORT =====\n")
    print(result["report"])
    print("\n===== CRITIC FEEDBACK =====\n")
    print(result["critic_feedback"])

if __name__ == "__main__":
    main()