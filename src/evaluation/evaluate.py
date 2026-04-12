from ..workflow.agent_pipeline import MultiAgentResearchSystem
from .baseline import SingleAgentBaseline
from .evaluator import Evaluator

def main():

    question = "What are the latest trends in renewable energy?"

    multi_agent = MultiAgentResearchSystem()
    baseline = SingleAgentBaseline()
    evaluator = Evaluator()

    print("\n--- Running Multi-Agent System ---")
    multi_result = multi_agent.run(question)
    multi_answer = multi_result["report"]

    print("\n--- Running Single-Agent Baseline ---")
    single_answer = baseline.run(question)

    print("\n--- Evaluating Multi-Agent Output ---")
    multi_eval = evaluator.evaluate(question, multi_answer)

    print("\n--- Evaluating Single-Agent Output ---")
    single_eval = evaluator.evaluate(question, single_answer)

    print("\n========== RESULTS ==========\n")

    print("MULTI-AGENT OUTPUT:\n")
    print(multi_answer)

    print("\nEVALUATION:\n")
    print(multi_eval)

    print("\n-----------------------------\n")

    print("SINGLE-AGENT OUTPUT:\n")
    print(single_answer)

    print("\nEVALUATION:\n")
    print(single_eval)


if __name__ == "__main__":
    main()