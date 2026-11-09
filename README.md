# Brazen Head: Ghost in the Machine

The software half of **the Brazen Head**, a talking brass oracle built for the Physical Computing module (Design Engineering, Imperial College London). Group 5: Sebastian Lawrence and Uthman Zakiuddin.

You feed the head tokens and ask it a question out loud. It "chews", wakes, and answers in a deep, metallic voice. Then it gives away its secret: the answer was never oracular wisdom, just **your own course notes**. It runs a small local LLM that answers only from the student's Obsidian vault, and it says which note the answer came from.

> *Mystify, then demystify.* The head plays on the mystique around LLMs, then reveals that your own past work did the thinking.

Everything runs **locally on one laptop**: no cloud, no API keys. The Bela inside the head handles the microphones, speaker, servos and LEDs, and talks to this server over Wi-Fi.

---

## How it works

```
 visitor speaks
      │
      ▼
 ┌──────────┐   WAV    ┌──────────────────────────── laptop (this repo) ───────────────────────────┐
 │  Bela    │ ───────► │ /listen   faster-whisper (CPU) ─► text                                     │
 │ (mics,   │          │ /ask      hybrid search over the vault ─► top passages ─► qwen3.5:4b (GPU) │
 │ speaker, │ ◄─────── │           ─► verdict + answer ─► reveal ("…your note on Metal Casting")    │
 │ servos)  │   WAV    │ /speak    Piper TTS (CPU) + "brass" effect ─► WAV                          │
 └──────────┘          └────────────────────────────────────────────────────────────────────────────┘
```

1. **Listen:** faster-whisper turns the question into text. It's given the note titles and a list of course jargon, so words like "Chvorinov" and "martensite" are spelled correctly.
2. **Search:** the question is matched against ~750 chunks of the vault in two ways: by meaning (`nomic-embed-text` embeddings) and by exact keywords (BM25). The two rankings are merged with reciprocal rank fusion, and the top 3 passages go to the model.
3. **Think:** `qwen3.5:4b` (via Ollama) first decides whether the passages actually answer the question. If they do, it answers in character, in plain spoken sentences with equations said in words. **One sentence per token fed**, up to 6.
4. **Refuse:** if the notes don't cover it, the head says so ("The Head knows not…"). Sometimes it points to the nearest note you *did* write. A similarity threshold alone can't make this call: "How does a jet engine work?" scored higher than real questions because a note mentions turbine blades.
5. **Reveal:** the code, not the model, names the note(s) the answer came from, so the source can never be made up.
6. **Speak:** Piper (`en_GB-alan-medium`) speaks the answer slightly deeper, through a comb-filter "inside a brass head" effect, one sentence at a time.

**Measured on an RTX PRO 500 (6 GB) laptop:** about **1.5 s** median from question text to finished answer. 12/12 on the test set: 6 answered from the right note, 6 off-topic questions refused.

---

## Setup (Windows)

**Needs:** Python 3.10+, [Ollama](https://ollama.com), an Obsidian vault of notes, and ideally an NVIDIA GPU with 6 GB or more.

```powershell
# 1. Models for Ollama
ollama pull qwen3.5:4b
ollama pull nomic-embed-text

# 2. Python environment (in this folder)
py -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt

# 3. Point it at your vault: edit VAULT in config.py, then build the search index
python indexer.py

# 4. Download the voice (or try several with: python try_voices.py)
python -m piper.download_voices en_GB-alan-medium --download-dir voices
```

The Whisper model (`base.en`, about 150 MB) downloads automatically the first time you listen.

---

## Using it

| Command | What it does |
|---|---|
| `python indexer.py` | Build or update the search index from the vault. Re-run after editing notes; only changed chunks are re-embedded. `--dry-run` shows the chunking. |
| `python search.py "what is chvorinov's rule"` | Test retrieval on its own. `--eval questions.txt` scores the whole test set. |
| `python ask.py "question" -t 3` | Ask in text, as if 3 tokens were fed. `--debug` shows the passages and raw reply; `--speak` says it aloud; `--eval questions.txt` runs the full test set with timings. |
| `python try_voices.py` | Hear candidate voices. `--pitch 1.0 0.9 0.85` compares how deep; `--no-brass` removes the effect. |
| `python listen.py` | Say a question and see what Whisper heard. `--devices` lists microphones. `--save q.wav` keeps the recording. |
| `python talk.py` | **The whole ritual on the laptop:** press Enter, ask aloud, hear the answer and the reveal. Reports the silence after you stop speaking (the gap the animation must cover). |
| `python server.py` | Run the server the Bela talks to (see below). |

### Test set

`questions.txt` holds `question | expected note` lines. Use `-` for questions that **aren't** in the notes, to check that the head refuses them. Aim for 15–20 questions across all modules.

---

## Server API (for the Bela / ESP32)

`python server.py` loads all the models, then prints the address to call, e.g. `http://192.168.1.23:8000`. Allow Python on **private** networks when Windows Firewall asks. Ctrl+C stops it and frees the GPU.

| Endpoint | Send | Get back |
|---|---|---|
| `GET /health` | | `{"ok": true, "notes": 158, "chunks": 745, ...}` |
| `POST /listen` | the question as a WAV file (any sample rate, mono or stereo) | `{"text": "...", "audio_seconds": 2.3, "transcribe_s": 0.6}` |
| `POST /ask` | `{"question": "...", "tokens": 3}` (tokens optional: defaults to `DEFAULT_TOKENS`) | `{"answerable": true, "answer": "...", "reveal": "...", "sources": [...], "timings": {...}}` |
| `POST /speak` | `{"text": "..."}` | `audio/wav`, the head's voice |

One visitor = `/listen` → `/ask` → `/speak` for the answer, then `/speak` for the reveal (the reveal is empty when the head refuses). Questions are handled one at a time.

PowerShell examples:

```powershell
Invoke-RestMethod http://localhost:8000/ask -Method Post -ContentType "application/json" `
    -Body '{"question": "what is chvorinov''s rule", "tokens": 2}'
Invoke-RestMethod http://localhost:8000/listen -Method Post -ContentType "audio/wav" -InFile q.wav
Invoke-WebRequest http://localhost:8000/speak -Method Post -ContentType "application/json" `
    -Body '{"text": "The Head knows not."}' -OutFile reply.wav
```

---

## Settings (`config.py`)

| Setting | Now | Meaning |
|---|---|---|
| `VAULT` | `C:\Users\sebjl\DesEng_Vault` | The Obsidian vault (read-only; only `Modules/` is indexed) |
| `CHAT_MODEL` / `EMBED_MODEL` | `qwen3.5:4b` / `nomic-embed-text` | Ollama models |
| `TOP_K`, `PASSAGE_MAX_WORDS` | 3, 250 | How much of the notes the model reads per question |
| `TEMPERATURE` | 0.2 | Lower = closer to the notes' wording (better for formulas) |
| `SIMILARITY_FLOOR` | 0.40 | Below this, the question is treated as nonsense without asking the model |
| `HINT_MIN_SIMILARITY` | 0.55 | A refusal only suggests a related note above this |
| `DEFAULT_TOKENS` | 2 | **Placeholder** token count until the break-beam counts real tokens |
| `SENTENCES_PER_TOKEN`, `MIN_/MAX_SENTENCES` | 1, 1, 6 | Answer length per token fed |
| `VOICE_MODEL` | `voices/en_GB-alan-medium.onnx` | Piper voice |
| `VOICE_PITCH` | 0.9 | Deeper voice (about 2 semitones down); pace is kept the same |
| `VOICE_LENGTH_SCALE`, `VOICE_SENTENCE_PAUSE` | 1.0, 0.3 s | Speaking pace |
| `VOICE_BRASS` | 0.6 | Metallic effect strength (0 = off, e.g. if the Bela adds its own) |
| `WHISPER_MODEL` | `base.en` | `tiny.en` faster, `small.en` more accurate |
| `WHISPER_EXTRA_WORDS` | course jargon | Add any word Whisper keeps mishearing |
| `LISTEN_SILENCE_SECONDS`, `LISTEN_MAX_SECONDS` | 1.2, 8 | When laptop-mic recording stops |
| `SERVER_PORT` | 8000 | |

**Models stay loaded only while the server runs:** every request asks Ollama to keep the model loaded (`keep_alive: -1`), and Ctrl+C unloads it. Don't set `OLLAMA_KEEP_ALIVE` globally.

---

## Troubleshooting

- **Every step takes ~2 s longer than expected:** use `http://127.0.0.1:11434`, not `localhost`. On Windows, `localhost` tries IPv6 first and stalls (already set in `config.py`).
- **The first question is slow:** that's a model loading. The server and `ask.py` warm both models up at start.
- **`ollama ps` doesn't show 100% GPU:** close other GPU-heavy apps and don't raise the context above 4096 on a 6 GB card.
- **"Heard nothing" / wrong mic:** `python listen.py --devices`, then set `LISTEN_DEVICE` in `listen.py`.
- **A formula comes out garbled:** lower `TEMPERATURE` further, or make that note's wording clearer.
- **The Bela can't reach the server:** check the firewall prompt was allowed on private networks, and that both are on the same network. The laptop's own hotspot is the most reliable option at a demo.

---

## Files

| File | Role |
|---|---|
| `config.py` | All settings |
| `vault_index.py` | Reading notes, chunking at bold headers, embeddings, hybrid search |
| `indexer.py`, `search.py` | Build the index; test retrieval |
| `head.py` | The head's "mind": prompt, verdict, refusals, reveal, speech-safe text |
| `voice.py`, `try_voices.py` | Piper TTS, pitch and brass effect; voice audition |
| `listen.py` | faster-whisper STT, WAV decoding, laptop-mic recording |
| `ask.py`, `talk.py` | Text and spoken testers |
| `server.py` | HTTP API for the Bela |
| `questions.txt` | Test set |

Not committed (see `.gitignore`): `.venv/`, `index/` (rebuilt by `indexer.py`), `voices/` (downloaded), `samples/`.

---

## Status and next steps

- [x] Retrieval, answering, refusals and reveal (tested with real models)
- [x] Voice (Piper) and listening (Whisper); full spoken loop working on the laptop
- [ ] Ritual protocol with the hardware: idle → tokens → chew → wake → listen → think → speak → reveal → sleep
- [ ] A single `/ritual` endpoint (audio + token count in, answer + reveal audio out) and a log of every exchange
- [ ] Wi-Fi test from an ESP32, then the Bela: recording, playback, jaw lip-sync, break-beam token count, head turn
- [ ] Demo reliability: start-up script, in-character fallback if the server is slow, laptop hotspot

## Built with

[Ollama](https://ollama.com) · [Qwen3.5](https://ollama.com/library/qwen3.5) · [nomic-embed-text](https://ollama.com/library/nomic-embed-text) · [Piper](https://github.com/OHF-Voice/piper1-gpl) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
