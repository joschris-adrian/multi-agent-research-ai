from .base_agent import BaseAgent


class CriticAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            role="Quality Reviewer",
            goal="Evaluate reports for clarity, accuracy, and completeness"
        )

    def review(self, report):

        prompt = f"""You are a critical reviewer tasked with finding weaknesses in this report.

        Be adversarial - assume the report has gaps and your job is to find them.

        Check for:
        - Claims made without supporting evidence
        - Important perspectives or counterarguments that are missing
        - Statistics cited without sources
        - Vague or ambiguous statements that need clarification
        - Any section that could mislead the reader

        Report to review:
        {report}

        List every issue you find. If the report is genuinely strong, say so but still suggest one improvement.
        """

        return self.run(prompt)