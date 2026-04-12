from .base_agent import BaseAgent


class WriterAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            role="Technical Writer",
            goal="Write a well-structured research report"
        )

    def write_report(self, insights):

        prompt = f"""
        Write a structured research report using the following insights.

        Format:

        Title
        Introduction
        Key Trends
        Industry Leaders
        Future Outlook
        Conclusion

        Insights:
        {insights}
        """

        return self.run(prompt)