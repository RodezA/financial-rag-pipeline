# Financial RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) pipeline over synthetic financial documents, built as a reference architecture demonstrating best practices across the full RAG stack. Every pipeline run is measurable via a built-in evaluation harness.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-rag-pipeline-4qwekobbbnuvmeyhvaq7bu.streamlit.app)

**Live demo:** [financial-rag-pipeline-4qwekobbbnuvmeyhvaq7bu.streamlit.app](https://financial-rag-pipeline-4qwekobbbnuvmeyhvaq7bu.streamlit.app) *(password-protected — contact for access)*

---

## Architecture

```
Documents → Chunking → Embeddings → Vector DB (Qdrant)
                                          │
Query → Embedding → Vector Search → Top-10 Docs
                                          │
                                    Cross-Encoder Reranker
                                          │
                                    Top-4 Chunks → Claude (claude-sonnet-4-6) → Answer
                                                                                    │
                                                                            Eval Harness
                                                                   precision / recall / faithfulness / quality
```

The reranker sits between vector search and the LLM — it operates on the Top-10 retrieval candidates to select the 4 most relevant chunks before generation. This is intentional: vector search is fast but coarse; the cross-encoder is slower but more semantically accurate, and runs on a small candidate set so cost is low.

---

## Stack

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | Qdrant (local Docker or Qdrant Cloud) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Claude `claude-sonnet-4-6` with prompt caching |
| Evaluation | Custom harness — LLM-as-judge via Claude |
| UI | Streamlit |
| Config | YAML-driven — no hardcoded model or DB choices |

---

## Evaluation Results

Evaluated on 20 synthetic financial Q&A pairs across three difficulty levels.

| Metric | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| Precision@4 | 0.92 | 0.81 | 0.68 | 0.80 |
| Recall@4 | 0.89 | 0.76 | 0.61 | 0.75 |
| Faithfulness (1–5) | 4.8 | 4.5 | 4.1 | 4.5 |
| Answer Quality (1–5) | 4.7 | 4.3 | 3.9 | 4.3 |
| Avg Latency | — | — | — | ~1,800ms |

> Results generated with reranker enabled. Re-run anytime: `PYTHONPATH=. python src/evaluation/run_eval.py`

**Reranker impact:** enabling the cross-encoder reranker improves Precision@4 by ~12 percentage points over vector search alone on this dataset.

---

## Project Structure

```
rag_pipeline/
├── src/
│   ├── ingestion/          # Document loading, chunking, embedding
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── pipeline.py     # Ingestion entry point
│   ├── retrieval/
│   │   └── vector_store.py # Qdrant client + similarity search
│   ├── reranking/
│   │   └── reranker.py     # Cross-encoder reranker
│   ├── generation/
│   │   └── llm_client.py   # Claude client with prompt caching
│   ├── evaluation/
│   │   ├── metrics.py      # precision, recall, faithfulness, quality
│   │   └── run_eval.py     # Eval harness entry point
│   ├── pipeline.py         # RAGPipeline orchestrator
│   └── app.py              # Streamlit UI
├── config/
│   └── config.yaml         # Model, DB, chunking, retrieval settings
├── data/
│   ├── raw/                # Synthetic financial documents (7 files)
│   └── eval/
│       └── qa_pairs.json   # 20 Q&A pairs with difficulty labels
└── tests/                  # Unit tests for each pipeline stage
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker Desktop (for local Qdrant)
- API keys: `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4. Ingest documents

```bash
PYTHONPATH=. python src/ingestion/pipeline.py
```

### 5. Launch the UI

```bash
PYTHONPATH=. python -m streamlit run src/app.py
```

---

## Configuration

All model and infrastructure choices live in `config/config.yaml` — nothing is hardcoded in source:

```yaml
embedding:
  model: "text-embedding-3-small"

retrieval:
  top_k: 10

reranking:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k_after_rerank: 4

generation:
  model: "claude-sonnet-4-6"
```

### No OpenAI key? Use a local embedding model

Swap `text-embedding-3-small` for `all-MiniLM-L6-v2` (runs on CPU, no API key):

1. Set `embedding.model: "all-MiniLM-L6-v2"` in `config/config.yaml`
2. In `src/ingestion/embedder.py` and `src/pipeline.py`, replace OpenAI embedding calls with `SentenceTransformer("all-MiniLM-L6-v2").encode(texts)`
3. In `src/retrieval/vector_store.py`, change `VECTOR_DIM` from `1536` to `384`
4. Re-run ingestion to rebuild the collection

### Qdrant Cloud vs local Docker

Set `qdrant.url` in `config/config.yaml` to your Qdrant Cloud cluster URL (and set `QDRANT_API_KEY` env var) for a persistent, always-on deployment. Leave `url: null` to use local Docker.

---

## Running Tests

```bash
PYTHONPATH=. python -m pytest tests/
```

---

## Prompt Caching

The Claude client uses `cache_control` on both the system prompt and retrieved context blocks. On repeated or similar queries this reduces input token costs and latency significantly — the Streamlit UI surfaces cache hit/miss token counts per query.

---

## Dataset

Seven synthetic financial documents covering quarterly earnings, annual reports, a deal memo, and a market outlook. All data is AI-generated and contains no real company information.
