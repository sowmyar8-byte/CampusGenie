"""CampusGenie RAG core: hybrid retrieval (dense + BM25), query rewriting, streaming generation."""
import json, math, os, re, time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = Path("index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.getenv("TOP_K", 4))
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.35))
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
USE_HYBRID = os.getenv("USE_HYBRID", "true").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
CANDIDATES = int(os.getenv("CANDIDATES", 10))
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention

FALLBACK = ("I can't confirm this from the university documents I have. "
            "Please contact the relevant university office or check the official notice.")

SYSTEM_PROMPT = (
    "You are CampusGenie, a friendly assistant for Dayananda Sagar University students. "
    "Answer ONLY from the context passages provided. If they do not contain the answer, "
    "reply exactly: I can't confirm this from the university documents I have. "
    "Never invent rules, numbers, dates or fees.\n"
    "Style: start with the direct answer in one sentence, then add short supporting "
    "details. Use plain, simple language. Use a short bullet list (lines starting with '- ') "
    "when there are several items or steps. Bold only key numbers or terms with **double asterisks**. "
    "Do NOT mention passage numbers, file names, page numbers or phrases like 'according to the context'. "
    "Keep it under 120 words. If the passages mention a caveat (for example rules differ by programme "
    "or year), include it in one short closing sentence."
)

REWRITE_PROMPT = (
    "Rewrite the student's latest message as ONE standalone question that makes sense without the "
    "conversation. Keep all names, numbers and technical terms. If it is already standalone, repeat it "
    "unchanged. Output ONLY the question, nothing else."
)

# ---------------------------------------------------------------- keyword search (BM25)
STOP = set("a an the is are was were of to in on for and or what how do does can i my me we it that "
           "this with be at by as if from when which who will should".split())


def tok(t):
    return [w for w in re.findall(r"[a-z0-9+]+", t.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(tok(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        n = len(docs)
        df = Counter()
        for c in self.docs:
            df.update(c.keys())
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q, out = tok(query), np.zeros(len(self.docs))
        for i, c in enumerate(self.docs):
            for w in q:
                if w in c:
                    f = c[w]
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


# ---------------------------------------------------------------- loading
_embedder = _reranker = _chunks = _vectors = _bm25 = None


def _load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)


def _load_index():
    global _chunks, _vectors, _bm25
    if _chunks is None:
        if not (INDEX_DIR / "chunks.json").exists():
            raise RuntimeError("Index not found. Run: python ingest.py")
        _chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        _vectors = np.load(INDEX_DIR / "embeddings.npy")
        _bm25 = BM25([c.get("title", "") + " " + c["text"] for c in _chunks])


def embed(texts):
    _load_embedder()
    return _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------- retrieval
def rerank(question, passages, k):
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    scores = _reranker.predict([(question, p["text"]) for p in passages])
    for p, sc in zip(passages, scores):
        p["rerank_score"] = float(sc)
    return sorted(passages, key=lambda p: -p["rerank_score"])[:k]


def retrieve(question, k=TOP_K, min_score=MIN_SCORE):
    """Dense search, optionally fused with BM25 (reciprocal rank fusion), optionally reranked.
    The semantic score threshold always decides whether a passage is trusted."""
    _load_index()
    dense = _vectors @ embed([QUERY_PREFIX + question])[0]
    pool = max(CANDIDATES, k) if (USE_HYBRID or USE_RERANKER) else k
    dense_rank = list(np.argsort(-dense)[:pool])
    if USE_HYBRID:
        bs = _bm25.scores(question)
        kw_rank = [i for i in np.argsort(-bs)[:pool] if bs[i] > 0]
        fused = {}
        for r, i in enumerate(dense_rank):
            fused[i] = fused.get(i, 0) + 1 / (60 + r)
        for r, i in enumerate(kw_rank):
            fused[i] = fused.get(i, 0) + 1 / (60 + r)
        order = sorted(fused, key=lambda i: -fused[i])
    else:
        order = dense_rank
    found = [dict(_chunks[i], score=float(dense[i])) for i in order if dense[i] >= min_score]
    if USE_RERANKER and found:
        found = rerank(question, found[:pool], k)
    return found[:k]


# ---------------------------------------------------------------- LLM calls
def complete(system, user, max_tokens=500):
    if PROVIDER == "anthropic":
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=max_tokens,
            temperature=0.2, system=system, messages=[{"role": "user", "content": user}])
        return msg.content[0].text.strip()
    r = requests.post(
        os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
        json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "stream": False,
              "options": {"temperature": 0.2, "num_predict": max_tokens},
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def stream_complete(system, user, max_tokens=500):
    """Yield the answer piece by piece."""
    if PROVIDER == "anthropic":
        import anthropic
        with anthropic.Anthropic().messages.stream(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=max_tokens,
                temperature=0.2, system=system, messages=[{"role": "user", "content": user}]) as s:
            yield from s.text_stream
        return
    with requests.post(
            os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
            json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "stream": True,
                  "options": {"temperature": 0.2, "num_predict": max_tokens},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            piece = d.get("message", {}).get("content", "")
            if piece:
                yield piece
            if d.get("done"):
                break


# ---------------------------------------------------------------- pipeline
def rewrite_question(history, question):
    """Turn a follow-up like 'and for labs?' into a standalone question using chat history."""
    if not history:
        return question
    convo = "\n".join(f"{'Student' if h['role'] == 'user' else 'Assistant'}: {h['content'][:300]}"
                      for h in history[-4:])
    try:
        out = complete(REWRITE_PROMPT, f"Conversation:\n{convo}\n\nLatest message: {question}\n\nStandalone question:", 80)
        out = out.strip().strip('"').splitlines()[0].strip() if out.strip() else ""
        return out if 3 < len(out) < 300 else question
    except Exception:
        return question


def build_prompt(question, passages):
    ctx = "\n\n".join(f"Passage {i} ({p['title']}):\n{p['text']}" for i, p in enumerate(passages, 1))
    return f"Context:\n{ctx}\n\nStudent question: {question}\n\nAnswer:"


def make_sources(passages):
    out, seen = [], set()
    for p in passages:
        if p["title"] in seen:
            continue
        seen.add(p["title"])
        out.append({"id": len(out) + 1, "title": p["title"], "file": p["file"], "url": p["url"],
                    "score": round(p["score"], 3),
                    "snippet": p["text"][:260] + ("..." if len(p["text"]) > 260 else "")})
    return out


def _abstained(text):
    t = text.lower()
    return "can't confirm" in t or "cannot confirm" in t


def stream_answer(question, history=None):
    """Yield ('rewrite'|'token'|'done', data) events."""
    t0 = time.time()
    standalone = rewrite_question(history or [], question)
    changed = standalone if standalone.strip().lower() != question.strip().lower() else ""
    if changed:
        yield "rewrite", changed
    passages = retrieve(standalone)
    if not passages:
        yield "done", {"answer": FALLBACK, "sources": [], "abstained": True,
                       "rewritten": changed, "latency_s": round(time.time() - t0, 2)}
        return
    text = ""
    for piece in stream_complete(SYSTEM_PROMPT, build_prompt(standalone, passages)):
        text += piece
        yield "token", piece
    text = text.strip()
    ab = _abstained(text)
    yield "done", {"answer": FALLBACK if ab else text, "sources": [] if ab else make_sources(passages),
                   "abstained": ab, "rewritten": changed, "latency_s": round(time.time() - t0, 2)}


def answer(question, history=None):
    """Non-streaming version (used by evaluate.py)."""
    final = {}
    for ev, data in stream_answer(question, history):
        if ev == "done":
            final = data
    return final
