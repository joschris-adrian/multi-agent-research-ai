from fastapi import FastAPI
from pydantic import BaseModel
from src.workflow.agent_pipeline import MultiAgentResearchSystem

app = FastAPI(
    title="Multi-Agent Research API",
    description="AI-powered research assistant using multiple agents",
    version="1.0"
)

# Initialize system once (important for performance)
system = MultiAgentResearchSystem()


# Request schema
class ResearchRequest(BaseModel):
    query: str


# Root endpoint
@app.get("/")
def home():
    return {"message": "Multi-Agent Research API is running"}


# Main research endpoint
@app.post("/research")
def research(request: ResearchRequest):

    result = system.run(request.query)

    return {
        "question": request.query,
        "tasks": result["tasks"],
        "insights": result["insights"],
        "report": result["report"],
        "critic_feedback": result["critic_feedback"]
    }