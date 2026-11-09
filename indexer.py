"""Build (or update) the search index from your vault.

    python indexer.py            # index the vault (only re-embeds chunks that changed)
    python indexer.py --rebuild  # re-embed everything
    python indexer.py --dry-run  # just show how the notes would be chunked; no Ollama needed

Run it again whenever you change your notes.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np

import config
from vault_index import DOC_PREFIX, chunk_note, embed, iter_note_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache and re-embed every chunk")
    ap.add_argument("--dry-run", action="store_true", help="chunk only; print stats, don't embed or save")
    args = ap.parse_args()

    if not config.VAULT.exists():
        sys.exit(f"Vault not found: {config.VAULT}  (edit VAULT in config.py)")

    t0 = time.time()
    chunks, notes = [], 0
    for path in iter_note_paths():
        cs = chunk_note(path)
        notes += 1
        chunks.extend(cs)
    sizes = [len(c["text"].split()) for c in chunks]
    print(f"{notes} notes -> {len(chunks)} chunks "
          f"(words per chunk: median {statistics.median(sizes):.0f}, max {max(sizes)}, min {min(sizes)})")

    if args.dry_run:
        by_note = {}
        for c in chunks:
            by_note.setdefault(c["note"], []).append(c)
        for note, cs in list(by_note.items())[:8]:
            print(f"\n{note}  [{cs[0]['module']}]")
            for c in cs:
                print(f"   {len(c['text'].split()):4d}w  {c['section'] or '(intro)'}")
        print("\n(dry run: nothing embedded or saved)")
        return

    out = config.INDEX_DIR
    out.mkdir(exist_ok=True)

    # Reuse vectors for chunks whose text hasn't changed since the last run.
    cache = {}
    if not args.rebuild and (out / "meta.json").exists():
        old_meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        if old_meta.get("embed_model") == config.EMBED_MODEL:
            old_chunks = json.loads((out / "chunks.json").read_text(encoding="utf-8"))
            old_vecs = np.load(out / "embeddings.npy")
            cache = {c["hash"]: v for c, v in zip(old_chunks, old_vecs)}

    todo = [i for i, c in enumerate(chunks) if c["hash"] not in cache]
    print(f"Embedding {len(todo)} new/changed chunks with {config.EMBED_MODEL} "
          f"({len(chunks) - len(todo)} reused)...")
    if todo:
        try:
            new = embed([chunks[i]["embed_text"] for i in todo], DOC_PREFIX)
        except OSError as e:
            sys.exit(f"Couldn't reach Ollama at {config.OLLAMA_URL}: {e}\n"
                     f"Is Ollama running, and have you pulled {config.EMBED_MODEL}?")
        dim = new.shape[1]
    else:
        dim = len(next(iter(cache.values())))
    vecs = np.zeros((len(chunks), dim), dtype=np.float32)
    for j, i in enumerate(todo):
        vecs[i] = new[j]
    for i, c in enumerate(chunks):
        if c["hash"] in cache:
            vecs[i] = cache[c["hash"]]

    np.save(out / "embeddings.npy", vecs)
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "embed_model": config.EMBED_MODEL,
        "vault": str(config.VAULT),
        "notes": notes,
        "chunks": len(chunks),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1), encoding="utf-8")
    print(f"Saved index to {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
