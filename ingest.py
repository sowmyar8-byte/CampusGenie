"""Read documents in data/, split into section-sized chunks, embed, save the index."""
import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader

import rag

MAX_WORDS, OVERLAP = 160, 25


def read_pages(path: Path):
    if path.suffix.lower() == ".pdf":
        for n, page in enumerate(PdfReader(str(path)).pages, 1):
            yield n, page.extract_text() or ""
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def parse_header(text: str):
    """Optional 'TITLE:' and 'SOURCE:' lines at the top of a document."""
    title = url = None
    body = []
    for line in text.splitlines():
        if line.startswith("TITLE:") and title is None:
            title = line[6:].strip()
        elif line.startswith("SOURCE:") and url is None:
            url = line[7:].strip()
        else:
            body.append(line)
    return title, url, "\n".join(body).strip()


def windows(words):
    step = MAX_WORDS - OVERLAP
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + MAX_WORDS])
        if i + MAX_WORDS >= len(words):
            break


def chunk(text: str):
    """Keep paragraphs (rules/sections) together; merge small ones; window huge ones."""
    cur, n = [], 0
    for p in (p.strip() for p in text.split("\n\n")):
        if not p:
            continue
        w = len(p.split())
        if w > MAX_WORDS:
            if cur:
                yield "\n\n".join(cur); cur, n = [], 0
            yield from windows(p.split())
            continue
        if cur and n + w > MAX_WORDS:
            yield "\n\n".join(cur); cur, n = [], 0
        cur.append(p); n += w
    if cur:
        yield "\n\n".join(cur)


def main():
    chunks = []
    files = [p for p in sorted(Path("data").iterdir()) if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not files:
        raise SystemExit("No documents found in data/")
    for f in files:
        for page, raw in read_pages(f):
            title, url, body = parse_header(raw)
            for c in chunk(body):
                chunks.append({"file": f.name, "title": title or f.stem.replace("_", " "),
                               "url": url or "", "page": page, "text": c})
    print(f"{len(files)} files -> {len(chunks)} chunks. Embedding...")
    vectors = rag.embed([f"{c['title']}\n{c['text']}" for c in chunks])
    rag.INDEX_DIR.mkdir(exist_ok=True)
    np.save(rag.INDEX_DIR / "embeddings.npy", vectors)
    (rag.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print("Index saved to index/")


if __name__ == "__main__":
    main()
