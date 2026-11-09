"""The Brazen Head's voice: Piper text-to-speech, plus an optional "brass" effect.

    from voice import Voice
    v = Voice()                          # loads the voice model (once)
    wav = v.speak_wav("Behold.")         # WAV file as bytes (for the Bela / a browser)
    for wav in v.speak_sentences(text):  # one WAV per sentence: start playing the first
        play(wav)                        #   while the rest is still being made
    v.play(wav)                          # play on this laptop (for testing)

Voices live in ./voices. Get one with:
    python -m piper.download_voices en_GB-alan-medium --download-dir voices
or run  python try_voices.py  to download several and hear each one.
"""
import io
import re
import wave
from pathlib import Path

import numpy as np

import config


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    """float32 mono samples in [-1, 1] -> 16-bit WAV file bytes."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _fft_convolve(x, h):
    n = len(x) + len(h) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[:n]


def brass(samples: np.ndarray, rate: int) -> np.ndarray:
    """A metallic, resonant 'voice from inside a brass head' effect.

    A comb filter (many quick, decaying echoes ~9 ms apart) gives a hollow, tinny ring,
    and a few longer, quieter echoes give a sense of an enclosed chamber.
    Tune it with VOICE_BRASS (0 = off, 1 = strong) in config.py.
    """
    amount = config.VOICE_BRASS
    if amount <= 0:
        return samples
    h = np.zeros(int(rate * 0.6), dtype=np.float32)
    h[0] = 1.0
    d = int(rate * 0.009)                     # comb: resonance at ~110 Hz and harmonics
    for k in range(1, 25):
        h[k * d] += (0.55 * amount) * (0.82 ** k)
    for t, g in ((0.071, 0.30), (0.113, 0.22), (0.167, 0.15), (0.241, 0.10)):  # chamber echoes
        h[int(rate * t)] += g * amount
    out = _fft_convolve(samples.astype(np.float32), h)
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.9).astype(np.float32)


def lower_pitch(samples: np.ndarray, factor: float) -> np.ndarray:
    """Deepen the voice: stretch the audio so it plays back lower (factor < 1).

    Like slowing down a tape: pitch AND speed drop together. The Voice class asks Piper
    to speak correspondingly faster, so the overall pace stays the same.
    """
    if factor >= 0.999 or len(samples) < 2:
        return samples
    n = int(len(samples) / factor)
    return np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class Voice:
    def __init__(self, model=None):
        from piper import PiperVoice, SynthesisConfig   # imported here so the rest works without piper
        path = Path(model or config.VOICE_MODEL)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        if not path.exists():
            raise FileNotFoundError(
                f"Voice model not found: {path}\n"
                f"Download one:  python -m piper.download_voices {path.stem} --download-dir voices\n"
                f"(or run  python try_voices.py)")
        self.name = path.stem
        self.voice = PiperVoice.load(str(path))
        self.rate = self.voice.config.sample_rate
        self.pitch = getattr(config, "VOICE_PITCH", 1.0)
        # Lowering the pitch also slows the audio by 1/pitch, so ask Piper to speak faster to compensate.
        self.syn = SynthesisConfig(length_scale=config.VOICE_LENGTH_SCALE * self.pitch,
                                   noise_scale=config.VOICE_NOISE_SCALE,
                                   volume=config.VOICE_VOLUME)

    def samples(self, text) -> np.ndarray:
        chunks = [c.audio_float_array for c in self.voice.synthesize(text, syn_config=self.syn)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        pause = np.zeros(int(self.rate * config.VOICE_SENTENCE_PAUSE), dtype=np.float32)
        joined = []
        for c in chunks:
            joined += [c.astype(np.float32), pause]
        return lower_pitch(np.concatenate(joined[:-1]), self.pitch)

    def speak_wav(self, text) -> bytes:
        """Whole text -> one WAV (bytes), with the brass effect applied."""
        return _wav_bytes(brass(self.samples(text), self.rate), self.rate)

    def speak_sentences(self, text):
        """Yield one WAV per sentence, so playback can start after the first."""
        for s in split_sentences(text):
            yield _wav_bytes(brass(self.samples(s), self.rate), self.rate)

    @staticmethod
    def play(wav: bytes):
        """Play a WAV on this computer and wait until it finishes (for testing)."""
        try:
            import winsound                     # built into Python on Windows
            winsound.PlaySound(wav, winsound.SND_MEMORY)
        except ImportError:
            Path("last_speech.wav").write_bytes(wav)
            print("  (not on Windows: saved to last_speech.wav)")
