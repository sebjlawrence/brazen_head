"""Shared code: read the vault, split notes into chunks, embed them, and search.

Used by indexer.py (builds the index) and search.py (tests retrieval).
Later the server imports `Index` from here to find passages for each question.
"""
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import config

# nomic-embed-text expects these task prefixes on its inputs.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


# ---------------------------------------------------------------- reading notes

FRONTMATTER = re.compile(r"\A(?:---\n.*?\n---\n)+", re.S)
HEADER = re.compile(r"^(?:\*\*(?P<bold>[^*\n]{1,80}?):?\*\*:?|#{1,6} (?P<hash>.+?))\s*$")
IMAGE = re.compile(r"!\[\[[^\]]*\]\]")
LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def iter_note_paths():
    for top in config.INCLUDE_DIRS:
        for path in sorted((config.VAULT / top).rglob("*.md")):
            rel = path.relative_to(config.VAULT)
            if not any(part in config.EXCLUDE_DIRS for part in rel.parts):
                yield path


def read_note(path: Path):
    """Return (frontmatter dict, body text) with links/images tidied for reading."""
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    meta = {}
    m = FRONTMATTER.match(text)
    if m:
        # Some notes have the block twice; the last one wins.
        for block in re.findall(r"---\n(.*?)\n---\n", m.group(0), re.S):
            try:
                meta.update(yaml.safe_load(block) or {})
            except yaml.YAMLError:
                pass
        text = text[m.end():]
    text = IMAGE.sub("", text)
    text = LINK.sub(lambda l: l.group(2) or l.group(1).split("/")[-1], text)
    return meta, text


# ---------------------------------------------------------------- chunking

def words(s):
    return len(s.split())


def split_sections(body):
    """Split a note body at its bold/# headers. Returns [(header, text)]."""
    sections, header, lines = [], "", []
    for line in body.split("\n"):
        h = HEADER.match(line.strip())
        if h:
            if any(l.strip() for l in lines):
                sections.append((header, "\n".join(lines).strip()))
            header, lines = (h.group("bold") or h.group("hash")).strip(), []
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((header, "\n".join(lines).strip()))
    def skip(h):
        h = h.lower().strip(": ")
        return any(h == s or h.startswith(s + "-") or h.startswith(s + " ") for s in config.SKIP_SECTIONS)
    return [(h, t) for h, t in sections if not skip(h)]


def split_blocks(text):
    """Break a long section into blocks: top-level bullets, or paragraphs."""
    blocks, cur = [], []
    for line in text.split("\n"):
        starts_block = (line.startswith("- ") or line.startswith("* ") or line.strip() == "")
        if starts_block and cur and any(c.strip() for c in cur):
            blocks.append("\n".join(cur).strip())
            cur = []
        if line.strip():
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip())
    # Hard-split anything still too long (e.g. a giant paragraph).
    out = []
    for b in blocks:
        w = b.split()
        while len(w) > config.CHUNK_MAX_WORDS:
            out.append(" ".join(w[: config.CHUNK_TARGET_WORDS]))
            w = w[config.CHUNK_TARGET_WORDS:]
        if w:
            out.append(b if len(w) == len(b.split()) else " ".join(w))
    return out


def pack(blocks):
    """Greedily pack blocks into chunks of about CHUNK_TARGET_WORDS."""
    chunks, cur = [], []
    for b in blocks:
        if cur and words("\n".join(cur + [b])) > config.CHUNK_TARGET_WORDS:
            chunks.append("\n".join(cur))
            cur = []
        cur.append(b)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_note(path: Path):
    meta, body = read_note(path)
    title = path.stem
    module = str(meta.get("module", path.relative_to(config.VAULT).parts[1]))
    summary = str(meta.get("summary", "")).strip()

    # Join tiny sections onto the next one so no chunk is just a heading.
    merged, carry_h, carry_t = [], [], []
    for header, text in split_sections(body):
        carry_h.append(header)
        carry_t.append(text)
        if words("\n".join(carry_t)) >= config.CHUNK_MIN_WORDS:
            merged.append((" / ".join(h for h in carry_h if h), "\n".join(carry_t)))
            carry_h, carry_t = [], []
    if carry_t:
        if merged:
            h, t = merged[-1]
            merged[-1] = (" / ".join(x for x in [h] + carry_h if x), t + "\n" + "\n".join(carry_t))
        else:
            merged.append((" / ".join(h for h in carry_h if h), "\n".join(carry_t)))

    out = []
    for header, text in merged:
        pieces = [text] if words(text) <= config.CHUNK_MAX_WORDS else pack(split_blocks(text))
        for i, piece in enumerate(pieces):
            section = header + (f" (part {i + 1})" if len(pieces) > 1 else "")
            context = f"{title} — {section}" if section else title
            embed_text = f"{context}\n{summary}\n\n{piece}"
            out.append({
                "note": title,
                "module": module,
                "path": str(path.relative_to(config.VAULT)),
                "section": section,
                "summary": summary,
                "text": piece,
                "embed_text": embed_text,
                "hash": hashlib.sha1(embed_text.encode("utf-8")).hexdigest(),
            })
    return out


# ---------------------------------------------------------------- Ollama

def ollama(endpoint, payload, timeout=300):
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(texts, prefix, keep_alive=None, batch=32):
    vecs = []
    for i in range(0, len(texts), batch):
        payload = {"model": config.EMBED_MODEL, "input": [prefix + t for t in texts[i:i + batch]]}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        vecs.extend(ollama("embed", payload)["embeddings"])
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)


# ---------------------------------------------------------------- keyword search

STOP = set("""a an and are as at be by for from has have how in is it its of on or that the
this to was what when where which who why will with does do can you your about into than
then there these those their them they we our explain describe define tell me""".split())


def tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(tokens(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = tokens(query)
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, (d, L) in enumerate(zip(self.docs, self.lens)):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * L / self.avg))
            out[i] = s
        return out


# ---------------------------------------------------------------- the index

class Index:
    def __init__(self, index_dir=None):
        d = Path(index_dir or config.INDEX_DIR)
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.vectors = np.load(d / "embeddings.npy")
        self.bm25 = BM25([c["embed_text"] for c in self.chunks])

    def search(self, question, k=None, keep_alive=None):
        """Hybrid search: embeddings + BM25, merged by reciprocal rank fusion."""
        k = k or config.TOP_K
        q = embed([question], QUERY_PREFIX, keep_alive=keep_alive)[0]
        dense = self.vectors @ q
        sparse = self.bm25.scores(question)
        rrf = np.zeros(len(self.chunks))
        for scores in (dense, sparse):
            for rank, i in enumerate(np.argsort(-scores)):
                rrf[i] += 1.0 / (60 + rank)
        top = np.argsort(-rrf)[:k]
        return [dict(self.chunks[i], score=float(rrf[i]), similarity=float(dense[i]),
                     keyword=float(sparse[i])) for i in top]
