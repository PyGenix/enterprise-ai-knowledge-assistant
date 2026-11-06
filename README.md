# Enterprise AI Knowledge Assistant

Python FastAPI RAG assistant for company policy Q&A.

Documents are chunked, embedded, and stored in PostgreSQL with pgvector. LangGraph connects retrieval to an LLM answer step and returns cited passages with each response.

## Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn
- **Orchestration:** LangChain / LangGraph
- **Vector store:** PostgreSQL + pgvector
- **LLM / embeddings:** OpenAI API
- **Runtime:** Docker Compose

## Features

- `POST /documents` ingest with source replace semantics
- `POST /ask` retrieval-augmented answers with ranked citations
- Sample policy documents for local demos
- One-command Docker Compose startup

## Quick start

```bash
cp .env.example .env
# set OPENAI_API_KEY in .env
docker compose up --build
```

API docs: http://localhost:8000/docs

## Author

Pooja Sharma — [Pooja101117](https://github.com/Pooja101117)
