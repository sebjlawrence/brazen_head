"""The Brazen Head server: the Bela/ESP32 sends questions here over Wi-Fi.

    python server.py

Endpoints (JSON):
    GET  /health          -> {"ok": true, "notes": 158, "chunks": 745, ...}
    POST /speak           <- {"text": "..."}            -> audio/wav (the Head's voice)
    POST /listen          <- WAV audio (body, Content-Type audio/wav)
                          -> {"text": "what is martensite", "audio_seconds": 2.1, "transcribe_s": 0.6}
    POST /ritual?tokens=2&rate=44100  <- WAV audio of the spoken question (body)
                          -> audio/wav: the spoken answer + reveal, mono 16-bit at `rate`
                             (headers: X-Heard, X-Answerable, X-Notes, X-Timings). Each exchange is
                             appended to ritual_log.jsonl.
    POST /ask             <- {"question": "what is martensite", "tokens": 3}
                          -> {"answerable": true, "answer": "...", "reveal": "...",
                              "sources": [{"note": ..., "section": ...}], "timings": {...}}

Test from another PowerShell window:
    Invoke-RestMethod http://localhost:8000/ask -Method Post -ContentType "application/json" `
        -Body '{"question": "what is chvorinov''s rule", "tokens": 2}'

The models are loaded when the server starts and stay loaded until you stop it (Ctrl+C),
which unloads them and frees the GPU.
"""
import json
import random
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np

import config
from head import Head

head = None
voice = None                    # loaded at start if a voice model is present (see try_voices.py)
ears = None                     # Whisper, loaded at start if faster-whisper is installed
ears_lock = threading.Lock()    # Whisper runs on the CPU, separately from the GPU lock
gpu_lock = threading.Lock()     # one question at a time: the GPU can't run two well

RITUAL_LOG = Path(__file__).parent / "ritual_log.jsonl"
NOT_HEARD = [
    "Speak up, mortal. The Head heard nothing.",
    "Your words were lost in the brass. Ask again.",
]
REVEAL_PAUSE = 0.7              # seconds of silence between the answer and the reveal


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  {self.client_address[0]}  {fmt % args}")

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {"ok": True, **head.index.meta})
        else:
            self._send(404, {"error": "not found"})

    def _send_wav(self, wav):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def _send_wav_with(self, wav, headers):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(wav)

    def _ritual(self, query):
        """Spoken question (WAV) in -> spoken answer + reveal (WAV) out. The whole pipeline in one call."""
        from listen import resample, to_wav
        from voice import brass
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if ears is None or voice is None:
            return self._send(503, {"error": "speech-to-text or voice not loaded (see the startup messages)"})
        if not body:
            return self._send(400, {"error": "send the WAV audio as the request body"})
        try:
            tokens = int(query.get("tokens", [config.DEFAULT_TOKENS])[0])
            rate = int(query.get("rate", [0])[0]) or voice.rate
        except ValueError:
            return self._send(400, {"error": "tokens and rate must be whole numbers"})
        t0 = time.time()
        try:                                            # keep the recording for checking levels/noise
            config.QUESTIONS_DIR.mkdir(exist_ok=True)
            (config.QUESTIONS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.wav").write_bytes(body)
            for old in sorted(config.QUESTIONS_DIR.glob("*.wav"))[:-config.SAVE_QUESTIONS]:
                old.unlink()
        except OSError:
            pass
        try:
            with ears_lock:
                heard = ears.transcribe(body)
        except Exception as e:
            return self._send(400, {"error": f"couldn't read the audio: {e}"})
        question = heard["text"].strip()
        t_heard = time.time()
        if len(question) < 2:
            r = {"answerable": False, "answer": random.choice(NOT_HEARD), "reveal": "", "sources": [],
                 "timings": {}}
        else:
            with gpu_lock:
                try:
                    r = head.ask(question, tokens=tokens)
                except OSError as e:
                    return self._send(503, {"error": f"Ollama unavailable: {e}"})
        t_answered = time.time()

        # Answer, a pause, then the reveal: one WAV at the Bela's sample rate.
        parts = [voice.samples(r["answer"])]
        if r.get("reveal"):
            parts += [np.zeros(int(voice.rate * REVEAL_PAUSE), np.float32), voice.samples(r["reveal"])]
        speech = brass(np.concatenate(parts), voice.rate)
        wav = to_wav(resample(speech, voice.rate, rate), rate)
        t_spoken = time.time()

        timings = {"audio_in_s": heard["audio_seconds"], "listen_s": round(t_heard - t0, 2),
                   "think_s": round(t_answered - t_heard, 2), "speak_s": round(t_spoken - t_answered, 2),
                   "total_s": round(t_spoken - t0, 2), "audio_out_s": round(len(speech) / voice.rate, 2)}
        notes = [s["note"] for s in r.get("sources", [])]
        print(f"  ritual: heard \"{question}\" [{tokens} tokens] -> "
              f"{'answer' if r['answerable'] else 'refusal'}; "
              f"listen {timings['listen_s']}s + think {timings['think_s']}s + speak {timings['speak_s']}s "
              f"= {timings['total_s']}s")
        try:
            with RITUAL_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "client": self.client_address[0],
                                    "heard": question, "tokens": tokens, "answerable": r["answerable"],
                                    "answer": r["answer"], "reveal": r.get("reveal", ""), "notes": notes,
                                    "timings": timings}, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return self._send_wav_with(wav, {
            "X-Heard": quote(question), "X-Answerable": str(r["answerable"]).lower(),
            "X-Notes": quote("; ".join(notes)), "X-Timings": json.dumps(timings)})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path.rstrip("/") == "/ritual":
            return self._ritual(parse_qs(url.query))
        if self.path.rstrip("/") == "/listen":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if ears is None:
                return self._send(503, {"error": "speech-to-text not loaded: pip install -r requirements.txt"})
            if not body:
                return self._send(400, {"error": "send the WAV audio as the request body"})
            try:
                with ears_lock:
                    r = ears.transcribe(body)
            except Exception as e:                   # bad/unsupported audio
                return self._send(400, {"error": f"couldn't read the audio: {e}"})
            print(f"  heard: \"{r['text']}\"  ({r['audio_seconds']}s audio, {r['transcribe_s']}s)")
            return self._send(200, r)
        if self.path.rstrip("/") == "/speak":
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                text = str(data["text"]).strip()
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": 'send JSON like {"text": "..."}'})
            if voice is None:
                return self._send(503, {"error": "no voice loaded: run python try_voices.py and set VOICE_MODEL"})
            t = time.time()
            wav = voice.speak_wav(text)
            print(f"  spoke {len(text.split())} words in {time.time() - t:.2f}s")
            return self._send_wav(wav)
        if self.path.rstrip("/") != "/ask":
            return self._send(404, {"error": "not found"})
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            question = str(data["question"]).strip()
            tokens = int(data.get("tokens", config.DEFAULT_TOKENS))  # placeholder until the Bela counts tokens
        except (ValueError, KeyError, TypeError):
            return self._send(400, {"error": 'send JSON like {"question": "...", "tokens": 2}'})
        if not question:
            return self._send(400, {"error": "empty question"})
        with gpu_lock:
            try:
                r = head.ask(question, tokens=tokens)
            except OSError as e:
                return self._send(503, {"error": f"Ollama unavailable: {e}"})
        r.pop("raw_reply", None)
        print(f"  Q: {question}  [{tokens} tokens] -> "
              f"{'answer' if r['answerable'] else 'refusal'} in {r['timings'].get('total_s', 0):.2f}s")
        self._send(200, r)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    global head
    try:
        head = Head()
    except FileNotFoundError:
        sys.exit("No index yet: run  python indexer.py  first.")
    print("Waking the Head (loading models onto the GPU)...", end=" ", flush=True)
    try:
        print(f"ready in {head.warm_up():.1f}s")
    except OSError as e:
        sys.exit(f"\nCouldn't reach Ollama at {config.OLLAMA_URL}: {e}")

    global voice, ears
    try:
        from listen import Ears
        print(f"Loading Whisper ({config.WHISPER_MODEL})...", end=" ", flush=True)
        ears = Ears()
        print("ready")
    except ImportError:
        print("\nNo speech-to-text yet (/listen disabled): pip install -r requirements.txt")
    try:
        from voice import Voice
        voice = Voice()
        print(f"Voice: {voice.name}")
    except (ImportError, FileNotFoundError) as e:
        print(f"No voice yet (/speak disabled): {str(e).splitlines()[0]}")

    server = ThreadingHTTPServer((config.SERVER_HOST, config.SERVER_PORT), Handler)
    print(f"Listening on http://localhost:{config.SERVER_PORT}  "
          f"(from the Bela: http://{lan_ip()}:{config.SERVER_PORT}/ritual)")
    print("Ctrl+C to stop and free the GPU.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("\nUnloading models...", end=" ", flush=True)
        head.unload()
        print("done. The Head sleeps.")


if __name__ == "__main__":
    main()
