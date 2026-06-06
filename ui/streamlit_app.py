import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests
import json

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Research Assistant", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — LLM provider configuration
# ---------------------------------------------------------------------------
from ui.provider_config import load_provider_config, save_provider_config, validate_gemini_key

with st.sidebar:
    st.header("LLM Provider")

    config = load_provider_config()
    current_provider = config["provider"]
    current_key = config["api_key"]

    provider = st.radio(
        "Select provider",
        options=["ollama", "gemini"],
        index=0 if current_provider == "ollama" else 1,
        format_func=lambda x: "Ollama (local)" if x == "ollama" else "Gemini (Google AI Studio)",
    )

    api_key_input = ""
    if provider == "gemini":
        st.caption("Analyst, writer and critic use gemini-2.5-flash. Planner, researcher and graph builder always use Ollama.")
        if current_key:
            st.success("API key stored.")
            if st.checkbox("Replace API key"):
                api_key_input = st.text_input("New Gemini API key", type="password")
        else:
            st.warning("No API key stored.")
            api_key_input = st.text_input("Gemini API key", type="password")
    else:
        st.caption("Runs fully locally. Ollama must be running with llama3.2 pulled.")

    if st.button("Save provider settings"):
        if provider == "gemini":
            key_to_save = api_key_input or current_key
            if not key_to_save:
                st.error("Please enter a Gemini API key.")
            else:
                with st.spinner("Validating API key..."):
                    valid, error = validate_gemini_key(key_to_save)
                if valid:
                    save_provider_config("gemini", key_to_save)
                    st.success("Saved. Takes effect on the next pipeline run.")
                else:
                    save_provider_config("ollama")
                    st.success("Saved. Takes effect on the next pipeline run.")
        else:
            save_provider_config("ollama")

    st.divider()
    st.caption(f"Active provider: **{current_provider}**")
    if current_provider == "gemini":
        remaining_note = "20 requests/day free tier (gemini-2.5-flash)"
        st.caption(remaining_note)

st.title("Multi-Agent Research Assistant")
st.caption("Six AI agents collaborate to research your question, retrieve academic papers, build a knowledge graph and write a structured report.")

question = st.text_input(
    "What do you want to research?",
    placeholder="e.g. What are the latest trends in renewable energy?"
)

if st.button("Run"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        AGENT_ICONS = {
            "planner": "🔍",
            "researcher": "🌐",
            "analyst": "🧠",
            "graph_builder": "🗺️",
            "writer": "✍️",
            "critic": "🔎"
        }

        AGENT_STEPS = {
            "planner": 1,
            "researcher": 2,
            "analyst": 3,
            "graph_builder": 4,
            "writer": 5,
            "critic": 6
        }

        progress = st.progress(0, text="Starting pipeline...")
        status = st.status("Running agents...", expanded=True)
        agent_slots = {}

        with status:
            for agent in ["planner", "researcher", "analyst", "graph_builder", "writer", "critic"]:
                agent_slots[agent] = st.empty()
                agent_slots[agent].write(f"{AGENT_ICONS[agent]} {agent.replace('_', ' ').title()} — waiting...")

        result = None

        try:
            with requests.post(
                f"{API_URL}/research/stream",
                json={"query": question},
                stream=True,
                timeout=900
            ) as response:
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        payload = json.loads(line[6:])
                        event = payload["event"]
                        data = payload["data"]

                        if event == "agent":
                            agent = data["agent"]
                            icon = AGENT_ICONS.get(agent, "▶")
                            step = AGENT_STEPS.get(agent, 1)
                            pct = int((step / 6) * 90)

                            if data["status"] == "running":
                                with status:
                                    agent_slots[agent].write(f"{icon} **{agent.replace('_', ' ').title()}** — {data['message']}")
                                progress.progress(pct, text=f"{agent.replace('_', ' ').title()} running...")

                            elif data["status"] == "done":
                                with status:
                                    agent_slots[agent].write(f"✅ **{agent.replace('_', ' ').title()}** — done")

                        elif event == "complete":
                            result = data
                            progress.progress(100, text="Done!")

        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API. Is `uvicorn api.main:app --reload` running?")
            st.stop()
        except requests.exceptions.ReadTimeout:
            st.error("Pipeline timed out. Try a simpler question or check server logs.")
            st.stop()

        if not result:
            st.error("Pipeline did not return a result.")
            st.stop()

        status.update(label="All agents completed", state="complete", expanded=False)
        st.success("Done!")

        st.subheader("Report")
        st.markdown(result["report"])
        st.download_button(
            label="Download report",
            data=result["report"],
            file_name=f"research_report_{question[:30].replace(' ', '_')}.md",
            mime="text/markdown"
        )

        entities = result.get("entities", {})
        if any(entities.values()):
            st.subheader("Knowledge Graph Entities")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Companies**")
                for c in entities.get("companies", []):
                    st.write(f"- {c}")
            with col2:
                st.markdown("**Trends**")
                for t in entities.get("trends", []):
                    st.write(f"- {t}")
            with col3:
                st.markdown("**Technologies**")
                for t in entities.get("technologies", []):
                    st.write(f"- {t}")
            st.caption(f"Query the graph at {API_URL}/graphql")

        with st.expander("Research tasks (planner output)"):
            st.write(result["tasks"])

        with st.expander("Extracted insights"):
            st.write(result["insights"])

        with st.expander("Critic feedback"):
            st.write(result["critic_feedback"])

        with st.expander("Documents retrieved"):
            docs = result.get("documents", [])
            if docs:
                for doc in docs:
                    st.markdown(f"**{doc.get('title', 'Untitled')}**")
                    st.caption(doc.get('source', ''))
                    st.write(doc.get('content', ''))
                    st.divider()
            else:
                st.write("No documents retrieved.")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"API docs: {API_URL}/docs")
        with col_b:
            st.caption(f"GraphQL: {API_URL}/graphql")
