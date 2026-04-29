from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearchAgent
from src.agents.analyst import AnalystAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent
from src.agents.graph_builder import GraphBuilderAgent
from src.graph.knowledge_graph import KnowledgeGraph


class MultiAgentResearchSystem:
    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self.graph_builder = GraphBuilderAgent()
        self.kg = KnowledgeGraph()

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

        print("\n[graph builder]")
        try:
            self.kg.add_topic(question)
            entities = self.graph_builder.extract_entities(insights, question)

            for company in entities.get("companies", []):
                self.kg.add_entity(company, "Company")
                self.kg.link_entity_to_topic(company, question)

            for trend in entities.get("trends", []):
                self.kg.add_entity(trend, "Trend")
                self.kg.link_entity_to_topic(trend, question)

            for tech in entities.get("technologies", []):
                self.kg.add_entity(tech, "Technology")
                self.kg.link_entity_to_topic(tech, question)

            for rel in entities.get("relationships", []):
                self.kg.link_entities(rel["source"], rel["target"], rel["relation"])

            print(f"stored {len(entities.get('companies', []))} companies, "
                  f"{len(entities.get('trends', []))} trends, "
                  f"{len(entities.get('technologies', []))} technologies")

        except Exception as e:
            print(f"[graph builder] neo4j unavailable, skipping: {e}")
            entities = {"companies": [], "trends": [], "technologies": [], "relationships": []}

        # writer now receives entities from the knowledge graph
        print("\n[writer]")
        report = self.writer.write_report(insights, entities)
        print(report)

        print("\n[critic]")
        feedback = self.critic.review(report)
        print(feedback)

        return {
            "question": question,
            "tasks": tasks,
            "documents": documents,
            "insights": insights,
            "entities": entities,
            "report": report,
            "critic_feedback": feedback,
        }
