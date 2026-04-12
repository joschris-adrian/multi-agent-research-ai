from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearchAgent
from src.agents.analyst import AnalystAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent


class MultiAgentResearchSystem:
    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    def run(self, question):
        print(f"\n[question] {question}")

        print("\n[planner]")
        tasks = self.planner.plan(question)
        print(tasks)

        print("\n[researcher]")
        search_query = self.researcher.extract_query(tasks, question)
        documents = self.researcher.search(search_query)
        print(f"got {len(documents)} documents")

        print("\n[analyst]")
        insights = self.analyst.analyze(documents, question)
        print(insights)

        print("\n[writer]")
        report = self.writer.write_report(insights)
        print(report)

        print("\n[critic]")
        feedback = self.critic.review(report)
        print(feedback)

        return {
            "question": question,
            "tasks": tasks,
            "documents": documents,
            "insights": insights,
            "report": report,
            "critic_feedback": feedback,
        }
