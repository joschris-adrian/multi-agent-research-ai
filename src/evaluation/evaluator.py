from ..agents.base_agent import BaseAgent


class Evaluator:

    def __init__(self):
        self.judge = BaseAgent(
            role="AI Evaluator",
            goal="Evaluate the quality of AI-generated responses"
        )

    def evaluate(self, question, answer):

        prompt = f"""
        Evaluate the following answer based on these criteria:

        1. Relevance (0-10)
        2. Completeness (0-10)
        3. Clarity (0-10)
        4. Accuracy (0-10)

        Provide a score for each and a short justification.

        Question:
        {question}

        Answer:
        {answer}

        Output format:

        Relevance: X/10
        Completeness: X/10
        Clarity: X/10
        Accuracy: X/10

        Explanation:
        """

        return self.judge.run(prompt)