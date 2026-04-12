from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            role="Task Planner",
            goal="Break complex research questions into step-by-step research tasks"
        )

    def plan(self, question):

        prompt = f"""
        Break the following research question into a list of clear tasks.

        Question:
        {question}

        Return a numbered task list.
        """

        return self.run(prompt)