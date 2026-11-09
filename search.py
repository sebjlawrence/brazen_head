"""Test retrieval on its own (no chat model involved).

    python search.py "what is chvorinov's rule"      # show the top chunks for one question
    python search.py --eval questions.txt            # score a whole test set

questions.txt format, one per line (lines starting with # are ignored):
    What is Chvorinov's rule? | Metal Casting
    How does a capacitor block DC? | Capacitors
    Who won the 2022 World Cup? | -          <- '-' means "not in my notes"
"""
import argparse
import sys
import time

import config
from vault_index import Index


def show(results):
    for r in results:
        print(f"  {r['score']:.4f}  sim {r['similarity']:.2f}  kw {r['keyword']:5.1f}   "
              f"{r['note']}  >  {r['section'] or '(intro)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--eval", metavar="FILE")
    ap.add_argument("-k", type=int, default=3, help="how many results count as a hit (default 3)")
    args = ap.parse_args()

    try:
        index = Index()
    except FileNotFoundError:
        sys.exit("No index yet: run  python indexer.py  first.")

    if args.question:
        t = time.time()
        results = index.search(args.question, k=max(args.k, config.TOP_K))
        print(f"\n{args.question}   ({(time.time() - t) * 1000:.0f} ms)\n")
        show(results)
        print("\nTop passage:\n" + "-" * 60 + "\n" + results[0]["text"][:1200])
        return

    if not args.eval:
        ap.print_help()
        return

    hits = total = 0
    for line in open(args.eval, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        q, expected = (s.strip() for s in line.split("|", 1))
        results = index.search(q, k=args.k)
        notes = [r["note"] for r in results]
        if expected == "-":
            print(f"  [ -- ] {q}\n         top sim {results[0]['similarity']:.2f} -> {notes[0]}  "
                  f"(should be low: not in notes)")
            continue
        total += 1
        ok = expected.lower() in (n.lower() for n in notes)
        hits += ok
        print(f"  [{'HIT ' if ok else 'MISS'}] {q}\n         expected {expected}; got {', '.join(notes)}")
    if total:
        print(f"\nHit rate (right note in top {args.k}): {hits}/{total} = {hits / total:.0%}")


if __name__ == "__main__":
    main()
