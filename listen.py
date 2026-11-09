"""The Brazen Head's ears: faster-whisper speech-to-text, plus microphone recording for testing.

    python listen.py                 # record a question from this laptop's mic and print the text
    python listen.py --file q.wav    # transcribe a WAV file
    python listen.py --devices       # list microphones (set LISTEN_DEVICE below if the wrong one is used)

In code:
    from listen import Ears, record
    ears = Ears()                            # loads Whisper once (~1-2 s)
    text = ears.transcribe(wav_bytes)["text"]  # WAV bytes (any sample rate) or a float32 16 kHz array

The Bela will record the visitor itself and send the WAV to the server's /listen endpoint.
"""
import argparse
import io
import json
import sys
import time
import wave

import numpy as np

import config

RATE = 16000                 # Whisper works at 16 kHz mono
LISTEN_DEVICE = None         # microphone index from --devices; None = Windows default


# ---------------------------------------------------------------- audio helpers

def wav_to_float(wav_bytes: bytes) -> np.ndarray:
    """Decode a WAV (8/16/24/32-bit, mono or stereo, any rate) to float32 mono at 16 kHz."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
             << 8 >> 8).astype(np.float32) / 8388608
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return resample(x, rate, RATE)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(x) < 2:
        return x.astype(np.float32)
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def to_wav(x: np.ndarray, rate: int = RATE) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------- vocabulary hint

def vocabulary_prompt(limit_chars=700):
    """A hint sentence of course words, so Whisper spells technical terms correctly."""
    words = [w.strip() for w in config.WHISPER_EXTRA_WORDS.split(",") if w.strip()]
    try:
        chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        for title in dict.fromkeys(c["note"] for c in chunks):
            if not any(ch.isdigit() for ch in title) and len(title) < 45:
                words.append(title)
    except (OSError, ValueError):
        pass                                        # no index yet: just use the extra words
    text = "Questions about engineering course notes: "
    for w in dict.fromkeys(words):
        if len(text) + len(w) + 2 > limit_chars:
            break
        text += w + ", "
    return text.rstrip(", ") + "."


# ---------------------------------------------------------------- transcription

class Ears:
    def __init__(self):
        from faster_whisper import WhisperModel      # imported here so the rest works without it
        self.model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                                  compute_type=config.WHISPER_COMPUTE)
        self.prompt = vocabulary_prompt()

    def transcribe(self, audio) -> dict:
        """audio: WAV bytes, or a float32 mono array at 16 kHz. Returns text, length and timing."""
        if isinstance(audio, (bytes, bytearray)):
            audio = wav_to_float(bytes(audio))
        seconds = len(audio) / RATE
        if seconds < 0.3:
            return {"text": "", "audio_seconds": round(seconds, 2), "transcribe_s": 0.0}
        t = time.time()
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=config.WHISPER_BEAM, initial_prompt=self.prompt,
            vad_filter=True,                  # skip silence (also stops Whisper "hearing" words in noise)
            condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text, "audio_seconds": round(seconds, 2), "transcribe_s": round(time.time() - t, 2)}


# ---------------------------------------------------------------- microphone recording (laptop testing)

def record(max_seconds=None, silence_seconds=None, start_timeout=None, device=None, on_start=None):
    """Record one question from the microphone.

    Waits for speech to start, then stops after a pause (LISTEN_SILENCE_SECONDS) or at
    LISTEN_MAX_SECONDS. Returns float32 16 kHz audio, or None if nobody spoke.
    """
    import sounddevice as sd                         # pip install sounddevice
    max_seconds = max_seconds or config.LISTEN_MAX_SECONDS
    silence_seconds = silence_seconds or config.LISTEN_SILENCE_SECONDS
    start_timeout = start_timeout or config.LISTEN_START_TIMEOUT
    block = int(RATE * 0.05)                         # 50 ms blocks

    blocks, levels = [], []
    started, quiet, waited = False, 0.0, 0.0
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=block,
                        device=device if device is not None else LISTEN_DEVICE) as stream:
        while True:
            data, _ = stream.read(block)
            x = data[:, 0].copy()
            level = float(np.sqrt(np.mean(x ** 2)))
            levels.append(level)
            # Background level from the first ~0.3 s; speech must be clearly louder.
            floor = float(np.median(levels[:6])) if len(levels) >= 6 else level
            threshold = max(floor * 3.0, 0.01)
            if not started:
                waited += 0.05
                blocks = (blocks + [x])[-6:]         # keep 0.3 s before speech starts
                if len(levels) > 6 and level > threshold:
                    started = True
                    if on_start:
                        on_start()
                elif waited > start_timeout:
                    return None
                continue
            blocks.append(x)
            quiet = quiet + 0.05 if level < threshold else 0.0
            if quiet >= silence_seconds or len(blocks) * 0.05 >= max_seconds:
                return np.concatenate(blocks)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="transcribe this WAV instead of recording")
    ap.add_argument("--devices", action="store_true", help="list audio input devices")
    ap.add_argument("--save", metavar="WAV", help="also save the recording")
    args = ap.parse_args()

    if args.devices:
        import sounddevice as sd
        print(sd.query_devices())
        print("\nSet LISTEN_DEVICE in listen.py to the number of the mic you want (None = default).")
        return

    print(f"Loading Whisper ({config.WHISPER_MODEL}, {config.WHISPER_DEVICE})...", end=" ", flush=True)
    t = time.time()
    try:
        ears = Ears()
    except ImportError:
        sys.exit("\nfaster-whisper isn't installed: pip install -r requirements.txt")
    print(f"ready in {time.time() - t:.1f}s")

    if args.file:
        audio = open(args.file, "rb").read()
    else:
        print("Speak your question...")
        audio = record(on_start=lambda: print("  (hearing you)"))
        if audio is None:
            sys.exit("Heard nothing. Check the microphone (python listen.py --devices).")
        if args.save:
            open(args.save, "wb").write(to_wav(audio))
    r = ears.transcribe(audio)
    print(f"\n  \"{r['text']}\"\n  [{r['audio_seconds']}s of audio, transcribed in {r['transcribe_s']}s]")


if __name__ == "__main__":
    main()
