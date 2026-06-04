from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            role="Task Planner",
            goal="Break complex research questions into step-by-step research tasks",
            gemini_model="gemini-2.5-flash"
        )

    def plan(self, question):

        prompt = f"""
        Break the following research question into a list of clear tasks.

        Example:
        Question: What are the latest trends in electric vehicles?
        Tasks:
        1. Search for recent EV market statistics and growth figures
        2. Identify major EV manufacturers and their latest models
        3. Research EV battery technology developments
        4. Find government policies and incentives affecting EV adoption

        Question:
        {question}

        Return a numbered task list.
        """

        return self.run(prompt)