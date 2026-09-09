# PDF RAG + Agent Backend

A Python backend that answers questions about your PDFs using Retrieval-
Augmented Generation, wrapped in a LangGraph Agent that can also reach for
a calculator tool when the question calls for math instead of search.

```
PDF -> Loader -> Chunk -> Gemini Embedding -> FAISS -> Retriever -> RAG Tool -> LangGraph Agent -> DeepSeek -> Final Answer
```

## 1. Architecture

```
                         User Question
                               |
                               v
                     +------------------+
                     |   DeepSeek LLM   |
                     |      Agent       |
                     +--------+---------+
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                RAG Tool             Calculator Tool
                    |
                    v
              Query Embedding (Gemini)
                    |
                    v
                 FAISS
                    |
                    v
              Relevant Chunks
                    |
                    v
              Agent State (LangGraph)
                    |
                    v
              DeepSeek LLM
                    |
                    v
               Final Answer
```

**Component responsibilities**

| Component | File | Responsibility |
|---|---|---|
| PDF Loader | `src/document_loader.py` | Read PDFs from `data/`, keep `source`/`page` metadata |
| Text Splitter | `src/text_splitter.py` | Chunk documents (`CHUNK_SIZE` / `CHUNK_OVERLAP`) |
| Embedding Service | `src/embeddings.py` | The *only* place that talks to the Gemini embedding API |
| Vector Store | `src/vector_store.py` | Build / save / load the local FAISS index |
| Retriever | `src/retriever.py` | Embed a question and run FAISS similarity search |
| RAG formatting | `src/rag.py` | Turn retrieved Documents into an LLM-ready context block + source list |
| Tools | `src/tools/` | `search_documents` (wraps the Retriever) and `calculator` |
| Agent | `src/agent/` | LangGraph state (`state.py`), nodes (`nodes.py`), graph (`graph.py`) |
| LLM | `src/llm.py` | The *only* place that talks to the DeepSeek chat API |
| Config | `src/config.py` | All tunables and API keys, read from `.env` |

## 2. Python version

Python 3.11+ (developed and tested on 3.12).

## 3. Installation

```bash
pip install -r requirements.txt
```

## 4. `.env` configuration

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your-gemini-key
DEEPSEEK_API_KEY=your-deepseek-key
```

`.env` is listed in `.gitignore` and must never be committed. Optional
overrides (chunk size, top-k, model names, etc.) are documented in
`.env.example` and read centrally by `src/config.py` — no other module
reads `os.environ` directly.

## 5. Importing PDFs

Drop one or more PDF files into `data/`:

```
data/
├── company_report.pdf
├── product_manual.pdf
└── policy.pdf
```

Then build the FAISS index:

```bash
python ingest.py
```

Expected output:

```
Found 3 PDF file(s) in .../data.
PDF files processed: 3
Chunks created: 428
FAISS index created successfully.
```

The index is written to `vectorstore/` and is loaded from disk on
subsequent runs — `ingest.py` only needs to be re-run when your PDFs
change.

## 6. Running the CLI

```bash
python main.py
```

```
PDF RAG Agent

Ask a question:
> What is the company's revenue?

Answer:
The company's revenue was ...

Sources:
- company_report.pdf, page 12

Ask a question:
> What is 125 * 37?

Answer:
4625

Ask a question:
> exit
Goodbye.
```

Behind the scenes, every question goes through the LangGraph agent:

1. **Agent node** — DeepSeek decides whether it needs a tool.
2. If the question is about the PDFs → it calls `search_documents`
   (Gemini-embeds the question, searches FAISS, returns matching chunks
   with `source`/`page`).
3. If the question is arithmetic → it calls `calculator`.
4. If neither applies (e.g. "Hello") → it answers directly.
5. Tool results are fed back to the agent node, which produces the
   final answer. `Agent.ask()` returns `{"answer": ..., "sources": [...]}`.

## 7. Programmatic usage

```python
from src.agent.graph import Agent

agent = Agent()
response = agent.ask("What is the main purpose of this document?")
print(response["answer"])
print(response["sources"])  # [] if no PDF search was needed
```

## 8. Running tests

```bash
pytest
```

The default suite (26 tests) requires **no API keys** — it uses a
deterministic fake embedding backend (`tests/conftest.py`) for FAISS/
retriever/tool tests, and pure unit tests for routing logic, so it
never fails just because a key is missing.

Tests that exercise the *real* Gemini/DeepSeek APIs are marked
skip-by-default and only run when you explicitly opt in:

```bash
RUN_LIVE_TESTS=true pytest
```

(this also requires `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` to be set in
`.env`, depending on which live tests apply).

Test files:

| File | Covers |
|---|---|
| `tests/test_loader.py` | PDF loading, source/page metadata |
| `tests/test_splitter.py` | Chunking, non-empty chunks, metadata |
| `tests/test_embeddings.py` | Embedding service construction + live Gemini calls |
| `tests/test_vector_store.py` | FAISS build/add/save/load/search |
| `tests/test_retriever.py` | Top-K, content, metadata |
| `tests/test_tools.py` | `search_documents` and `calculator` |
| `tests/test_agent.py` | Tool-call routing + live agent tool selection |
| `tests/test_e2e.py` | Full loader→chunk→embed→FAISS→retriever pipeline, plus a live full pipeline test |

## 9. HTTP API (for a frontend UI)

An optional FastAPI layer lives in `src/api/` on top of the same
`Agent` / `run_ingestion()` used by the CLI — no business logic is
duplicated.

Install the extra dependencies (already in `requirements.txt` /
`pyproject.toml`) and run:

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

Interactive docs (Swagger UI) are then available at
`http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Index/key status, for the frontend to show setup prompts |
| `POST` | `/chat` | `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |
| `POST` | `/ingest` | Re-run ingestion on everything in `data/`, rebuild the in-memory Agent |
| `POST` | `/documents/upload` | Multipart PDF upload into `data/` (does not auto-ingest) |

`CORSMiddleware` in `src/api/app.py` currently allows
`http://localhost:3000` — update `allow_origins` for your actual
frontend's dev/prod URLs.

The API is stateless/single-turn per request, same as `Agent.ask()` —
there's no multi-turn conversation memory yet. Adding that would mean
a `session_id` on `ChatRequest` and a checkpointer (e.g. LangGraph's
`SqliteSaver`) keyed by session.

## 10. Project structure

```
pdf-rag-agent/
├── data/
│   └── sample.pdf
├── vectorstore/
├── src/
│   ├── config.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── ingestion.py       # shared ingestion logic (used by ingest.py and POST /ingest)
│   ├── tools/
│   │   ├── search_tool.py
│   │   └── calculator_tool.py
│   ├── agent/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── graph.py
│   ├── api/               # optional FastAPI layer for a frontend UI
│   │   ├── app.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── llm.py
│   └── rag.py
├── tests/
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_splitter.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_tools.py
│   ├── test_agent.py
│   └── test_e2e.py
├── ingest.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 11. Agent workflow (LangGraph)

```
START
  |
  v
Agent  <---------------------+
  |                          |
  need tool?                 |
  ├── No  --> END            |
  |                          |
  └── Yes                    |
        |                    |
        v                    |
    Tool Node ----------------+
```

State (`src/agent/state.py`):

```python
{
    "messages": [...],            # LangGraph message history (accumulated)
    "question": "...",
    "retrieved_documents": [...], # populated only when search_documents runs
    "tool_results": [...],
    "final_answer": "...",
}
```

## Notes

- `EmbeddingService` and the DeepSeek client (`src/llm.py`) are the only
  places that construct their respective API clients — everything else
  (retriever, tools, agent) depends on those abstractions, not on
  Gemini/DeepSeek directly, so the provider is swappable and easy to
  mock in tests.
- Adding a new tool later (web search, DB query, ...) only means adding
  a file under `src/tools/` and including it in `Agent.__init__`'s tool
  list — no changes needed elsewhere.
<!-- 小结命令表：

操作	命令
建索引	uv run python ingest.py
问答 CLI	uv run python main.py
跑测试	uv run pytest -q -->


<!-- 填入你的两个 key：

env
GEMINI_API_KEY=你的gemini密钥
DEEPSEEK_API_KEY=你的deepseek密钥

2. 放入 PDF 文件

把你要问答的 PDF 放进 data/ 文件夹（项目自带了一个 sample.pdf 用于测试）：

data/
└── sample.pdf

3. 建立向量索引

powershell
uv run python ingest.py

正常输出：

Found 1 PDF file(s) in ...\data.
PDF files processed: 1
Chunks created: 12
FAISS index created successfully.

这一步会调用 Gemini API 做 embedding，生成的索引存在 vectorstore/ 里，PDF 不变的话只需要跑一次。

4. 启动对话 CLI

powershell
uv run python main.py
PDF RAG Agent

Ask a question:
> What is the company's revenue?

Answer:
...

Sources:
- sample.pdf, page 2

输入 exit 退出。

5.（可选）跑测试确认环境没问题

powershell
uv run pytest -q -->