"""
A2A (Agent-to-Agent) server — exposes each agent as an independent
FastAPI endpoint so agents can call each other dynamically rather than
running in a fixed sequence.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearchAgent
from src.agents.analyst import AnalystAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent
from src.agents.graph_builder import GraphBuilderAgent

app = FastAPI()

planner = PlannerAgent()
researcher = ResearchAgent()
analyst = AnalystAgent()
writer = WriterAgent()
critic = CriticAgent()
graph_builder = GraphBuilderAgent()


class PlanRequest(BaseModel):
    question: str


class ResearchRequest(BaseModel):
    query: str
    max_results: int = 3


class AnalyzeRequest(BaseModel):
    documents: list
    query: str


class WriteRequest(BaseModel):
    insights: str
    entities: Optional[dict] = {}


class ReviewRequest(BaseModel):
    report: str


class ExtractRequest(BaseModel):
    insights: str
    topic: str


@app.post("/agent/planner")
def plan(request: PlanRequest):
    try:
        result = planner.plan(request.question)
        return {"result": result}
    except Exception as e:
        return {"error": str(e), "result": ""}


@app.post("/agent/researcher")
def research(request: ResearchRequest):
    try:
        docs = researcher.search(request.query, max_results=request.max_results)
        return {"result": docs}
    except Exception as e:
        return {"error": str(e), "result": []}


@app.post("/agent/analyst")
def analyze(request: AnalyzeRequest):
    try:
        insights = analyst.analyze(request.documents, request.query)
        return {"result": insights}
    except Exception as e:
        return {"error": str(e), "result": ""}


@app.post("/agent/writer")
def write(request: WriteRequest):
    try:
        report = writer.write_report(request.insights, request.entities)
        return {"result": report}
    except Exception as e:
        return {"error": str(e), "result": ""}


@app.post("/agent/critic")
def review(request: ReviewRequest):
    try:
        feedback = critic.review(request.report)
        return {"result": feedback}
    except Exception as e:
        return {"error": str(e), "result": ""}


@app.post("/agent/graph_builder")
def extract(request: ExtractRequest):
    try:
        entities = graph_builder.extract_entities(request.insights, request.topic)
        return {"result": entities}
    except Exception as e:
        return {"error": str(e), "result": {}}