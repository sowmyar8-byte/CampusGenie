# CampusGenie 🎓
An AI-powered university student assistant using Retrieval-Augmented Generation (RAG).
Students ask questions in plain language; the system retrieves passages from approved university
documents and an LLM answers **only** from them, with sources and a fallback when unsure.

**Flow:** question → embedding → semantic search → top passages → LLM prompt → answer + sources

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env           # Windows: copy .env.example .env
```

### Choose an LLM (in `.env`)
- **Free/local:** install [Ollama](https://ollama.com), then `ollama pull llama3.2` and keep `LLM_PROVIDER=ollama`.
- **Claude API:** set `LLM_PROVIDER=anthropic` and add `ANTHROPIC_API_KEY`.

## Run
```bash
# 1. Put approved PDFs / .txt / .md files in data/  (a sample FAQ is included)
python ingest.py      # builds the index
python app.py         # open http://127.0.0.1:5000
python evaluate.py    # runs tests.json -> results.csv
```

## Structure
| File | Purpose |
|---|---|
| `ingest.py` | extract text, chunk (300 words, 50 overlap), embed, save index |
| `rag.py` | retrieval with similarity threshold, prompt, LLM call, fallback |
| `app.py` | Flask backend (`/`, `/ask`) |
| `templates/index.html` | chat UI |
| `evaluate.py`, `tests.json` | evaluation harness |

## Notes
- Index only **approved, non-confidential** documents. Answers are AI-generated; verify with official sources.
- Tune `MIN_SCORE` in `.env` if it abstains too often (lower it) or answers off-topic questions (raise it).
- `data/` holds a knowledge base written from the official DSU MCA Handbook, the DSU library page and the MCA Brochure 2026 (paraphrased, each file has a `SOURCE:` link). `data_sample/` holds fictional sample policies, not indexed.
- Add your own documents as `.txt` with optional first lines `TITLE: ...` and `SOURCE: https://...`, or as PDFs, then run `python ingest.py`.
- **Reranker (optional):** set `USE_RERANKER=true` in `.env`. The top `CANDIDATES` passages are re-scored by a cross-encoder and the best `TOP_K` are sent to the LLM. Run `evaluate.py` with it off and on and compare the results.

## Advanced features
- **Hybrid retrieval:** dense embeddings + BM25 keyword search merged with reciprocal rank fusion (`USE_HYBRID`).
- **Follow-up memory:** the last turns of the chat are used to rewrite follow-up questions into standalone queries.
- **Streaming answers** over server-sent events (`/ask_stream`).
- **Voice input** (browser speech recognition) and **feedback logging** (`feedback.jsonl`).
- Optional **cross-encoder reranker** (`USE_RERANKER`).
