import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

for _key in ("GROQ_API_KEY", "QDRANT_API_KEY"):
    if _key in st.secrets and not os.environ.get(_key):
        os.environ[_key] = st.secrets[_key]

from src.pipeline import RAGPipeline

st.set_page_config(page_title="Financial RAG", layout="wide")


st.title("Financial Document Assistant")
st.caption("Grounded answers from synthetic financial documents — RAG + Llama 3.3")

with st.sidebar:
    st.header("Pipeline Settings")
    use_reranker = st.toggle("Enable reranker", value=True)
    st.markdown("---")
    st.markdown("**Model:** llama-3.3-70b-versatile")
    st.markdown("**Embeddings:** text-embedding-3-small")
    st.markdown("**Reranker:** ms-marco-MiniLM-L-6-v2")
    st.markdown("**Vector DB:** Qdrant")
    st.markdown("---")
    st.markdown("**Documents indexed:**")
    docs = [
        "novatech_q1_2024.txt",
        "novatech_q2_2024.txt",
        "meridian_capital_q1_2024.txt",
        "apex_holdings_annual_2023.txt",
        "cornerstone_bank_q3_2024.txt",
        "market_outlook_q3_2024.txt",
        "greenfield_infra_deal_memo.txt",
    ]
    for d in docs:
        st.markdown(f"- {d}")


@st.cache_resource(show_spinner="Loading pipeline models...")
def get_pipeline(reranker: bool) -> RAGPipeline:
    return RAGPipeline(use_reranker=reranker)


pipeline = get_pipeline(use_reranker)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])

question = st.chat_input("Ask a question about the financial documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Querying..."):
            result = pipeline.query(question)

        st.write(result["answer"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Latency", f"{result['latency_ms']:.0f}ms")
        col2.metric("Input tokens", result["token_usage"]["input_tokens"])
        col3.metric("Output tokens", result["token_usage"]["output_tokens"])

        with st.expander(f"Sources ({len(result['source_docs'])} chunks)"):
            for i, chunk in enumerate(result["source_docs"]):
                retrieval_score = (
                    result["retrieval_scores"][i]
                    if i < len(result["retrieval_scores"])
                    else None
                )
                rerank_score = (
                    result["rerank_scores"][i]
                    if result["rerank_scores"] and i < len(result["rerank_scores"])
                    else None
                )
                score_str = f"retrieval: `{retrieval_score:.3f}`" if retrieval_score is not None else ""
                if rerank_score is not None:
                    score_str += f" | rerank: `{rerank_score:.3f}`"
                st.markdown(f"**[{i+1}] {chunk['source']}** — {score_str}")
                st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                if i < len(result["source_docs"]) - 1:
                    st.divider()

    st.session_state.history.append(
        {"question": question, "answer": result["answer"]}
    )
