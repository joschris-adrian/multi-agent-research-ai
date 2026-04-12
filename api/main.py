import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from strawberry.fastapi import GraphQLRouter
from src.workflow.agent_pipeline import MultiAgentResearchSystem
from src.graphql.graphql_schema import schema

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
def research(request: ResearchRequest):
    result = system.run(request.query)
    return {
        "question": request.query,
        "tasks": result["tasks"],
        "insights": result["insights"],
        "entities": result["entities"],
        "report": result["report"],
        "critic_feedback": result["critic_feedback"]
    }
