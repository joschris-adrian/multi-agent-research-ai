from .base_agent import BaseAgent


class CriticAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            role="Quality Reviewer",
            goal="Evaluate reports for clarity, accuracy, and completeness"
        )

    def review(self, report):

        prompt = f"""
        Review the following report.

        Evaluate:

        - factual accuracy
        - clarity
        - missing information
        - logical flow

        Suggest improvements if needed.

        Report:
        {report}
        """

        return self.run(prompt)