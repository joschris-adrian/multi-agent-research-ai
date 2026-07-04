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
                    st.success("Saved. Takes effect immediately on the next query.")
                else:
                    save_provider_config("ollama")
                    st.success("Saved. Takes effect immediately on the next query.")
        else:
            save_provider_config("ollama")

    st.divider()
    st.caption(f"Active provider: **{current_provider}**")
    if current_provider == "gemini":
        remaining_note = "20 requests/day free tier (gemini-2.5-flash)"
        st.caption(remaining_note)

    # ---------------------------------------------------------------------------
    # Sidebar — Subscription management
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("Subscriptions")

    # Fetch current subscriptions
    try:
        subs_resp = requests.get(f"{API_URL}/subscriptions", timeout=5)
        current_subs = subs_resp.json() if subs_resp.status_code == 200 else []
    except Exception:
        current_subs = []
        st.caption("⚠️ Could not reach API to load subscriptions.")

    # Run Now button
    if st.button("▶ Run Scheduler Now", use_container_width=True, type="primary"):
        with st.spinner("Triggering scheduler..."):
            try:
                requests.post(f"{API_URL}/subscriptions/run", timeout=10)
                st.success("Scheduler triggered!")
            except Exception as e:
                st.error(f"Failed to trigger: {e}")

    # Create new subscription form
    with st.expander("➕ Create New Subscription", expanded=False):
        with st.form("new_sub_form"):
            new_topic = st.text_input("Topic", placeholder="e.g. AI agents")
            new_freq = st.selectbox("Frequency", ["daily", "weekly", "monthly"])
            new_method = st.selectbox("Delivery Method", ["log", "email", "slack", "discord"])
            new_target = st.text_input("Delivery Target", placeholder="email or webhook URL (not needed for 'log')")
            submitted = st.form_submit_button("Create Subscription")
            
            if submitted:
                if not new_topic.strip():
                    st.error("Topic is required.")
                elif new_method != "log" and not new_target.strip():
                    st.error("Delivery target is required for this method.")
                else:
                    try:
                        resp = requests.post(f"{API_URL}/subscriptions", json={
                            "topic": new_topic,
                            "frequency": new_freq,
                            "delivery_method": new_method,
                            "delivery_target": new_target
                        }, timeout=5)
                        if resp.status_code == 200:
                            st.success("Subscription created!")
                            st.rerun()
                        else:
                            st.error(f"Failed to create: {resp.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to API: {e}")

    # Display existing subscriptions
    if not current_subs:
        st.info("No subscriptions yet.")
    else:
        for sub in current_subs:
            status_text = "Paused" if sub.get("paused") else "Active"
            last_run_text = sub.get("last_run") or "Never"
            icon = "⏸️" if sub.get("paused") else "✅"
            
            with st.expander(f"{icon} {sub.get('topic', 'No topic')} ({status_text})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Frequency:** {sub.get('frequency', '?').capitalize()}")
                    st.write(f"**Method:** {sub.get('delivery_method', '?').capitalize()}")
                with col2:
                    st.write(f"**Last Run:** {last_run_text}")
                    if sub.get("delivery_target"):
                        st.write(f"**Target:** `{sub.get('delivery_target')}`")
                
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    # Toggle Pause/Resume
                    btn_label = "▶ Resume" if sub.get("paused") else "⏸ Pause"
                    if st.button(btn_label, key=f"toggle_{sub['id']}"):
                        endpoint = "resume" if sub.get("paused") else "pause"
                        try:
                            requests.post(f"{API_URL}/subscriptions/{sub['id']}/{endpoint}", timeout=5)
                            st.rerun()
                        except Exception:
                            st.error("API unavailable")
                
                with bcol2:
                    if st.button("🗑 Delete", key=f"del_{sub['id']}"):
                        try:
                            requests.delete(f"{API_URL}/subscriptions/{sub['id']}", timeout=5)
                            st.rerun()
                        except Exception:
                            st.error("API unavailable")

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