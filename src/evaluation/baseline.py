from ..agents.base_agent import BaseAgent


class SingleAgentBaseline:

    def __init__(self):
        self.agent = BaseAgent(
            role="General AI Assistant",
            goal="Answer questions directly"
        )

    def run(self, question):

        prompt = f"""
        Answer the following question in a detailed and structured way:

        {question}
        """

        return self.agent.run(prompt)