import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from strawberry.fastapi import GraphQLRouter
from src.workflow.agent_pipeline import MultiAgentResearchSystem
from src.graphql.graphql_schema import schema
from src.graph.knowledge_graph import KnowledgeGraph

from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

import json



app = FastAPI(
    title="Multi-Agent Research API",
    description="AI-powered research assistant with knowledge graph and GraphQL",
    version="2.0"
)

system = MultiAgentResearchSystem()

# mount GraphQL at /graphql
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


class ResearchRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Multi-Agent Research API is running"}


@app.post("/research")
async def research(request: ResearchRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, system.run, request.query)
    return {
        "question": request.query,
        "tasks": result["tasks"],
        "documents": result["documents"],
        "insights": result["insights"],
        "entities": result["entities"],
        "report": result["report"],
        "critic_feedback": result["critic_feedback"]
    }

@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    async def generate():
        import asyncio
        from src.agents.planner import PlannerAgent
        from src.agents.researcher import ResearchAgent
        from src.agents.analyst import AnalystAgent
        from src.agents.graph_builder import GraphBuilderAgent
        from src.agents.writer import WriterAgent
        from src.agents.critic import CriticAgent

        def send(event, data):
            return f"data: {json.dumps({'event': event, 'data': data})}\n\n"

        loop = asyncio.get_event_loop()
        question = request.query

        yield send("start", {"message": f"Starting pipeline for: {question}"})

        planner = PlannerAgent()
        yield send("agent", {"agent": "planner", "status": "running", "message": "Breaking question into tasks..."})
        tasks = await loop.run_in_executor(None, planner.plan, question)
        yield send("agent", {"agent": "planner", "status": "done", "output": tasks})

        researcher = ResearchAgent()
        yield send("agent", {"agent": "researcher", "status": "running", "message": "Searching web and arXiv..."})
        search_query = await loop.run_in_executor(None, researcher.extract_query, tasks, question)
        documents = await loop.run_in_executor(None, researcher.search, search_query)
        yield send("agent", {"agent": "researcher", "status": "done", "output": f"Found {len(documents)} documents"})

        analyst = AnalystAgent()
        yield send("agent", {"agent": "analyst", "status": "running", "message": "Retrieving from memory and extracting insights..."})
        insights = await loop.run_in_executor(None, analyst.analyze, documents, question)
        yield send("agent", {"agent": "analyst", "status": "done", "output": insights[:200] + "..."})

        graph_builder = GraphBuilderAgent()
        kg = KnowledgeGraph()
        yield send("agent", {"agent": "graph_builder", "status": "running", "message": "Extracting entities for knowledge graph..."})
        try:
            kg.add_topic(question)
            entities = await loop.run_in_executor(None, graph_builder.extract_entities, insights, question)
            for company in entities.get("companies", []):
                kg.add_entity(company, "Company")
                kg.link_entity_to_topic(company, question)
            for trend in entities.get("trends", []):
                kg.add_entity(trend, "Trend")
                kg.link_entity_to_topic(trend, question)
            for tech in entities.get("technologies", []):
                kg.add_entity(tech, "Technology")
                kg.link_entity_to_topic(tech, question)
            for rel in entities.get("relationships", []):
                kg.link_entities(rel["source"], rel["target"], rel["relation"])
        except Exception as e:
            entities = {"companies": [], "trends": [], "technologies": [], "relationships": []}
        yield send("agent", {"agent": "graph_builder", "status": "done", "output": f"Found {len(entities.get('companies', []))} companies, {len(entities.get('trends', []))} trends"})

        writer = WriterAgent()
        yield send("agent", {"agent": "writer", "status": "running", "message": "Generating structured report..."})
        report = await loop.run_in_executor(None, writer.write_report, insights, entities)
        yield send("agent", {"agent": "writer", "status": "done", "output": report[:200] + "..."})

        critic = CriticAgent()
        yield send("agent", {"agent": "critic", "status": "running", "message": "Reviewing report for gaps..."})
        feedback = await loop.run_in_executor(None, critic.review, report)
        yield send("agent", {"agent": "critic", "status": "done", "output": feedback[:200] + "..."})

        yield send("complete", {
            "question": question,
            "tasks": tasks,
            "documents": documents,
            "insights": insights,
            "entities": entities,
            "report": report,
            "critic_feedback": feedback
        })

    return StreamingResponse(generate(), media_type="text/event-stream")
