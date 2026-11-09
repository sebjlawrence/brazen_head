"""Download a few candidate voices and hear the Brazen Head in each.

    python try_voices.py                 # download + play every candidate
    python try_voices.py --no-brass      # same, without the metallic effect
    python try_voices.py en_US-ryan-high # just one voice
    python try_voices.py en_GB-alan-medium --pitch 1.0 0.9 0.85 0.8   # compare how deep

Samples are also saved in ./samples so you can replay them. Then set VOICE_MODEL in config.py.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import config

CANDIDATES = [
    "en_GB-alan-medium",                    # British male, calm
    "en_GB-northern_english_male-medium",   # British male, northern accent, deeper
    "en_US-ryan-high",                      # US male, rich, high quality
    "en_US-joe-medium",                     # US male, low and steady
    "en_GB-semaine-medium",                 # British, several speakers, theatrical
]
LINE = ("Who wakes the Brazen Head? Speak your question, mortal. "
        "Engineering stress equals the force divided by the original area. "
        "No oracle spoke here. You wrote this, in your note on Mechanical Properties of Materials.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voices", nargs="*", default=CANDIDATES)
    ap.add_argument("--no-brass", action="store_true")
    ap.add_argument("--pitch", type=float, nargs="+", help="play each voice at these pitch settings")
    args = ap.parse_args()
    if args.no_brass:
        config.VOICE_BRASS = 0

    here = Path(__file__).parent
    (here / "voices").mkdir(exist_ok=True)
    (here / "samples").mkdir(exist_ok=True)
    from voice import Voice

    for name in args.voices:
        model = here / "voices" / f"{name}.onnx"
        if not model.exists():
            print(f"Downloading {name}...")
            r = subprocess.run([sys.executable, "-m", "piper.download_voices", name,
                                "--download-dir", str(here / "voices")])
            if r.returncode != 0 or not model.exists():
                print(f"  couldn't download {name}, skipping")
                continue
        for pitch in (args.pitch or [config.VOICE_PITCH]):
            config.VOICE_PITCH = pitch
            v = Voice(model)
            wav = v.speak_wav(LINE)
            out = here / "samples" / f"{name}-pitch{pitch}{'' if args.no_brass else '-brass'}.wav"
            out.write_bytes(wav)
            print(f"Playing {name} at pitch {pitch}  (saved to {out.name})")
            v.play(wav)

    print("\nPick one and set VOICE_MODEL (and VOICE_PITCH) in config.py")


if __name__ == "__main__":
    main()
