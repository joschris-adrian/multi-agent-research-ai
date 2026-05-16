from ..agents.base_agent import BaseAgent


class Evaluator:

    def __init__(self):
        self.judge = BaseAgent(
            role="AI Evaluator",
            goal="Evaluate the quality of AI-generated responses",
            max_tokens=1000
        )

    def evaluate(self, question, answer):

        prompt = f"""
        Evaluate the following answer based on these four criteria.
        You MUST provide a numeric score for each criterion before any explanation.
        Do NOT skip any criterion. Do NOT write introductory sentences before the scores.

        Question:
        {question}

        Answer:
        {answer}

        Respond in exactly this format and no other:

        Relevance: X/10
        Completeness: X/10
        Clarity: X/10
        Accuracy: X/10

        Explanation:
        [2-3 sentences justifying the scores above]
        """

        return self.judge.run(prompt)