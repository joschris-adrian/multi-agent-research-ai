"""
A2A pipeline — replaces the fixed sequential pipeline with a dynamic
coordinator that calls agents via HTTP. The analyst can request
additional research mid-analysis if insights are insufficient.
"""

import httpx
from .a2a_client import A2AClient
from src.graph.knowledge_graph import KnowledgeGraph

AGENT_BASE_URL = "http://localhost:8004"


class A2AResearchSystem:
    def __init__(self):
        self.client = A2AClient()
        self.kg = KnowledgeGraph()

    def _is_insufficient(self, insights):
        if not insights:
            return True
        weak_phrases = [
            "no information",
            "insufficient",
            "unable to find",
            "no data",
            "not enough"
        ]
        return any(phrase in insights.lower() for phrase in weak_phrases)

    def run(self, question):
        print(f"\n[a2a] question: {question}")

        # planner
        print("\n[a2a] calling planner...")
        tasks = self.client.call_agent("planner", {"question": question})
        print(tasks)

        # researcher
        print("\n[a2a] calling researcher...")
        documents = self.client.call_agent("researcher", {
            "query": question,
            "max_results": 3
        }) or []
        print(f"  got {len(documents)} documents")

        # analyst — with dynamic re-research if insights are weak
        print("\n[a2a] calling analyst...")
        insights = self.client.call_agent("analyst", {
            "documents": documents,
            "query": question
        })

        if self._is_insufficient(insights):
            print("\n[a2a] insights insufficient — requesting additional research...")
            extra_docs = self.client.call_agent("researcher", {
                "query": f"{question} detailed analysis",
                "max_results": 3
            }) or []
            documents = documents + extra_docs
            insights = self.client.call_agent("analyst", {
                "documents": documents,
                "query": question
            })
            print(f"  re-analysed with {len(documents)} total documents")

        print(insights)

        # graph builder
        print("\n[a2a] calling graph builder...")
        entities = self.client.call_agent("graph_builder", {
            "insights": insights,
            "topic": question
        }) or {}

        try:
            self.kg.add_topic(question)
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
                self.kg.link_entities(
                    rel["source"], rel["target"], rel["relation"]
                )
        except Exception as e:
            print(f"[a2a] neo4j unavailable: {e}")

        # writer
        print("\n[a2a] calling writer...")
        report = self.client.call_agent("writer", {
            "insights": insights,
            "entities": entities
        })
        if not report:
            print("[a2a] writer failed — using insights as fallback report")
            report = f"## Summary\n{insights}\n\n## Conclusion\nReport generation failed — insights above represent the research findings."
        print(report)

        print("\n[a2a] calling critic...")
        feedback = self.client.call_agent("critic", {"report": report})
        if not feedback:
            feedback = "Critic unavailable — manual review recommended."
        print(feedback)

        if feedback and any(phrase in feedback.lower() for phrase in [
            "missing", "incorrect", "inaccurate", "unclear", "vague"
        ]):
            print("\n[a2a] critic flagged issues — requesting improved report...")
            revised = self.client.call_agent("writer", {
                "insights": f"{insights}\n\nCritic feedback to address:\n{feedback}",
                "entities": entities
            })
            if revised:
                report = revised
                feedback = self.client.call_agent("critic", {"report": report}) or feedback
                print("[a2a] report revised based on critic feedback")

        return {
            "question": question,
            "tasks": tasks,
            "documents": documents,
            "insights": insights,
            "entities": entities,
            "report": report,
            "critic_feedback": feedback,
        }