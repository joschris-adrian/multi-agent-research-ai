import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")
st.caption("Five AI agents collaborate to research your question, build a knowledge graph, and write a structured report.")

question = st.text_input(
    "What do you want to research?",
    placeholder="e.g. What are the latest trends in renewable energy?"
)

if st.button("Run"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Agents working..."):
            try:
                response = requests.post(
                    f"{API_URL}/research",
                    json={"query": question},
                    timeout=300
                )
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is `uvicorn api.main:app --reload` running?")
                st.stop()

        st.success("Done!")

        st.subheader("Report")
        st.write(result["report"])

        # knowledge graph entities
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
