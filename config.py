"""Settings for the Brazen Head server. Edit these, not the code."""
import os
from pathlib import Path

# Your Obsidian vault. Read-only: nothing here is ever written to.
VAULT = Path(os.environ.get("BRAZEN_VAULT", r"C:\Users\sebjl\DesEng_Vault"))

# Only folders under the vault listed here are indexed.
INCLUDE_DIRS = ["Modules"]

# Any path containing one of these folder names is skipped.
EXCLUDE_DIRS = {"Archive", "Inbox", "Projects", "Maps", "attachments", ".obsidian", ".git"}

# Trailing note sections that aren't course content.
SKIP_SECTIONS = {"related", "references"}

# Ollama
# 127.0.0.1, not "localhost": on Windows, "localhost" tries IPv6 first and waits ~2 s per request.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3.5:4b"

# Chunking (in words). The embedding model reads about 2048 tokens (~1500 words) at most.
CHUNK_TARGET_WORDS = 350   # aim for chunks about this long
CHUNK_MAX_WORDS = 500      # never go over this
CHUNK_MIN_WORDS = 60       # sections shorter than this are joined to the next one

# Where the index is saved (inside this project folder, not the vault).
INDEX_DIR = Path(__file__).parent / "index"

# Search
TOP_K = 3                  # chunks passed to the LLM per question (fewer = faster; the right note is nearly always #1)
PASSAGE_MAX_WORDS = 250    # each passage is trimmed to this before going to the LLM (shorter prompt = faster reply)
SIMILARITY_FLOOR = 0.40    # below this the question is treated as nonsense without asking the LLM.
                           # NOT the main off-topic test: the LLM decides that (see head.py).
HINT_MIN_SIMILARITY = 0.55 # a refusal only suggests a related note ("ask me instead of...") above this

# Answering
TEMPERATURE = 0.2          # lower = more literal, higher = more theatrical
KEEP_ALIVE = -1            # keep models loaded while the server runs (-1 = forever, 0 = unload now)
# --- Tokens -> answer length ---
# PLACEHOLDER: until the IR break-beam counts real tokens, any request that doesn't
# send a token count uses this. The Bela will send the real count in its /ask request.
DEFAULT_TOKENS = 2
SENTENCES_PER_TOKEN = 1    # each fed token buys this many sentences...
MIN_SENTENCES = 1          # ...at least this many (even for 0 tokens)
MAX_SENTENCES = 6          # ...and at most this many

# Voice (Piper text-to-speech). Run  python try_voices.py  to hear the candidates.
VOICE_MODEL = "voices/en_GB-alan-medium.onnx"   # path relative to this folder
VOICE_PITCH = 0.9          # 1.0 = natural; lower = deeper (0.9 is about 2 semitones down).
                           # Speaking pace is kept the same automatically. Below ~0.75 it sounds slowed-down.
VOICE_LENGTH_SCALE = 1.0   # >1 = slower, more solemn; 1.0 = normal speed
VOICE_NOISE_SCALE = 0.5    # lower = flatter, more mechanical; Piper's default is about 0.667
VOICE_VOLUME = 1.0
VOICE_SENTENCE_PAUSE = 0.3   # seconds of silence between sentences
VOICE_BRASS = 0.6          # metallic "inside a brass head" effect: 0 = off, 1 = strong
                           # (set to 0 once the Bela does its own effects)

# Listening (faster-whisper speech-to-text). Runs on the CPU so it doesn't compete with
# Ollama for GPU memory. The model downloads automatically the first time (about 150 MB for base.en).
WHISPER_MODEL = "base.en"  # "tiny.en" = fastest, "base.en" = good balance, "small.en" = more accurate, slower
WHISPER_DEVICE = "cpu"     # "cuda" is faster, but needs NVIDIA's cuBLAS/cuDNN libraries installed
WHISPER_COMPUTE = "int8"   # int8 is fastest on a CPU
WHISPER_BEAM = 5           # 1 = fastest; 5 = more accurate
# Course words Whisper might otherwise mishear. Your note titles are added automatically.
WHISPER_EXTRA_WORDS = ("Chvorinov, martensite, austenite, ferrite, cementite, pearlite, eutectic, "
                       "eutectoid, Poisson's ratio, Young's modulus, sintering, thermoset, thermoplastic, "
                       "elastomer, galvanic, Laplace, Fourier, eigenvalue, eigenvector, Kirchhoff, "
                       "Thevenin, capacitor, impedance, Arduino, ESP32, MicroPython")

# Recording from the laptop microphone (listen.py, talk.py). The Bela does its own recording.
LISTEN_MAX_SECONDS = 8       # longest question allowed
LISTEN_SILENCE_SECONDS = 1.2 # stop this long after the speaker goes quiet
LISTEN_START_TIMEOUT = 6     # give up if nobody starts speaking within this time

# Server (the Bela/ESP32 calls this over Wi-Fi)
SERVER_HOST = "0.0.0.0"    # listen on the network, not just this laptop
SERVER_PORT = 8000
