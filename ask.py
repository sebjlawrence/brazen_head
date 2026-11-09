"""Talk to the Brazen Head from the terminal.

    python ask.py                          # interactive: type questions, Ctrl+C to quit
    python ask.py "what is chvorinov's rule" -t 3
    python ask.py --eval questions.txt     # run the full pipeline over the test set
    python ask.py --debug                  # also show the passages and the model's raw reply
    python ask.py --speak                  # ...and hear the Head say it (needs a voice: see try_voices.py)

-t / --tokens  how many tokens were "fed" (sets the answer length; default is DEFAULT_TOKENS in config.py)
"""
import argparse
import sys
import time

import config
from head import Head


def show(r, debug):
    print(f"\n  HEAD ({'answers' if r['answerable'] else 'refuses'}, {r['sentences']} sentence max):")
    print(f"  {r['answer']}")
    if r["reveal"]:
        print(f"  REVEAL: {r['reveal']}")
    t = r["timings"]
    print(f"  [search {t.get('search_s', 0):.2f}s, llm {t.get('llm_s', 0):.2f}s, "
          f"total {t.get('total_s', t.get('search_s', 0)):.2f}s"
          + (f", model load {t['model_load_s']:.1f}s" if t.get("model_load_s", 0) > 0.5 else "") + "]")
    if debug:
        for p in r["passages"]:
            print(f"     sim {p['similarity']:.2f}  {p['note']} > {p['section'] or '(intro)'}")
        if r.get("raw_reply"):
            print("  --- raw reply ---\n" + r["raw_reply"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-t", "--tokens", type=int, default=config.DEFAULT_TOKENS)
    ap.add_argument("--eval", metavar="FILE")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--speak", action="store_true", help="speak answers aloud with Piper")
    args = ap.parse_args()

    try:
        head = Head()
    except FileNotFoundError:
        sys.exit("No index yet: run  python indexer.py  first.")
    print("Waking the Head (loading models)...", end=" ", flush=True)
    try:
        print(f"ready in {head.warm_up():.1f}s")
    except OSError as e:
        sys.exit(f"\nCouldn't reach Ollama: {e}. Is it running?")

    voice = None
    if args.speak:
        from voice import Voice
        try:
            voice = Voice()
        except FileNotFoundError as e:
            sys.exit(str(e))
        print(f"Voice: {voice.name}")

    def answer(q, tokens):
        r = head.ask(q, tokens=tokens)
        show(r, args.debug)
        if voice:
            t = time.time()
            first = True
            for wav in voice.speak_sentences(r["answer"]):
                if first:
                    print(f"  [first sentence ready to speak after {time.time() - t:.2f}s of synthesis]")
                    first = False
                voice.play(wav)
            if r["reveal"]:
                voice.play(voice.speak_wav(r["reveal"]))
        return r

    if args.eval:
        rows = []
        for line in open(args.eval, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            q, expected = (s.strip() for s in line.split("|", 1))
            r = head.ask(q, tokens=args.tokens) if not voice else None
            if voice:
                print(f"\n> {q}")
                r = answer(q, args.tokens)
            if expected == "-":
                ok = not r["answerable"]
            else:
                ok = r["answerable"] and any(s["note"].lower() == expected.lower() for s in r["sources"])
            rows.append((ok, r["timings"].get("total_s", 0)))
            print(f"\n[{'PASS' if ok else 'FAIL'}] {q}   (expected: {expected})")
            if not voice:
                show(r, args.debug)
        passed = sum(ok for ok, _ in rows)
        times = sorted(t for _, t in rows)
        print(f"\n{passed}/{len(rows)} passed. Time per question: median {times[len(times) // 2]:.2f}s, "
              f"worst {times[-1]:.2f}s")
        return

    if args.question:
        answer(args.question, args.tokens)
        return

    print("Ask the Head (Ctrl+C to quit). Prefix with a number to set tokens, e.g. '3 what is martensite'")
    try:
        while True:
            q = input("\n> ").strip()
            if not q:
                continue
            tokens = args.tokens
            first, _, rest = q.partition(" ")
            if first.isdigit() and rest:
                tokens, q = int(first), rest
            answer(q, tokens)
    except (KeyboardInterrupt, EOFError):
        print("\nThe Head falls silent.")


if __name__ == "__main__":
    main()
