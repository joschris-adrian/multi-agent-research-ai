import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.workflow.agent_pipeline import MultiAgentResearchSystem

def main():
    question = input("Enter a research question: ")
    system = MultiAgentResearchSystem()
    result = system.run(question)
    print("\n===== FINAL REPORT =====\n")
    print(result["report"])
    print("\n===== CRITIC FEEDBACK =====\n")
    print(result["critic_feedback"])

if __name__ == "__main__":
    main()