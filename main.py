from src.workflow.crew_pipeline import run_multi_agent_system


def main():

    question = input("Enter a research question: ")

    result = run_multi_agent_system(question)

    print("\n\n===== FINAL RESULT =====\n")
    print(result)


if __name__ == "__main__":
    main()