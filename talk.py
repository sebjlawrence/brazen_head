"""Talk to the Brazen Head out loud, on this laptop: listen -> think -> speak -> reveal.

    python talk.py            # press Enter, ask your question aloud, hear the answer
    python talk.py -t 4       # as if 4 tokens were fed (longer answers)
    python talk.py --quiet    # print the answer instead of speaking it

This is the whole ritual minus the hardware. On the real Head, the Bela records and
plays the audio, and talks to server.py instead.
"""
import argparse
import sys
import time

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, default=config.DEFAULT_TOKENS)
    ap.add_argument("--quiet", action="store_true", help="don't speak, just print")
    args = ap.parse_args()

    from head import Head
    from listen import Ears, record

    print("Waking the Head...")
    t = time.time()
    try:
        head = Head()
    except FileNotFoundError:
        sys.exit("No index yet: run  python indexer.py  first.")
    try:
        head.warm_up()
    except OSError as e:
        sys.exit(f"Couldn't reach Ollama: {e}")
    try:
        ears = Ears()
    except ImportError:
        sys.exit("faster-whisper isn't installed: pip install -r requirements.txt")
    voice = None
    if not args.quiet:
        from voice import Voice
        try:
            voice = Voice()
        except (ImportError, FileNotFoundError) as e:
            print(f"(no voice, printing instead: {str(e).splitlines()[0]})")
    from head import sentences_for
    print(f"Ready in {time.time() - t:.1f}s.  Tokens: {args.tokens} "
          f"(answers up to {sentences_for(args.tokens)} sentence(s))")

    try:
        while True:
            input("\nPress Enter, then ask your question aloud (Ctrl+C to quit)...")
            print("  listening...")
            audio = record(on_start=lambda: print("  (hearing you)"))
            if audio is None:
                print("  Heard nothing.")
                continue
            heard = ears.transcribe(audio)
            q = heard["text"]
            print(f"  You asked: \"{q}\"   [{heard['audio_seconds']}s of speech, "
                  f"transcribed in {heard['transcribe_s']}s]")
            if not q:
                print("  (couldn't make out any words)")
                continue

            r = head.ask(q, tokens=args.tokens)
            think = r["timings"].get("total_s", r["timings"].get("search_s", 0))
            print(f"  HEAD: {r['answer']}")
            if r["reveal"]:
                print(f"  REVEAL: {r['reveal']}")
            if voice:
                t = time.time()
                for i, wav in enumerate(voice.speak_sentences(r["answer"])):
                    if i == 0:
                        print(f"  [thinking {think:.2f}s + first sentence {time.time() - t:.2f}s; "
                              f"silence after you stopped: about "
                              f"{heard['transcribe_s'] + think + time.time() - t:.1f}s]")
                    voice.play(wav)
                if r["reveal"]:
                    voice.play(voice.speak_wav(r["reveal"]))
            else:
                print(f"  [silence after you stopped: about {heard['transcribe_s'] + think:.1f}s]")
    except (KeyboardInterrupt, EOFError):
        print("\nThe Head falls silent.")


if __name__ == "__main__":
    main()
