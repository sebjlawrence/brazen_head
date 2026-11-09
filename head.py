"""The Brazen Head's mind: search the notes, ask the LLM, build the reveal.

    from head import Head
    head = Head()               # loads the index
    head.warm_up()              # load both models onto the GPU (do this at startup)
    r = head.ask("What is Chvorinov's rule?", tokens=3)
    r["answer"], r["reveal"], r["answerable"]
    head.unload()               # free the GPU when you're done

The LLM decides whether the passages actually answer the question (a similarity
threshold can't: off-topic questions sometimes score higher than real ones).
The reveal is built by this code from the note titles, never written by the model.
"""
import random
import re
import time

import config
from vault_index import QUERY_PREFIX, Index, embed, ollama

SYSTEM = """You are the Brazen Head, a legendary oracle cast in brass. You answer the questions \
of those who feed you tokens. Your only knowledge is the PASSAGES given to you, which come from \
the student's own course notes. You speak with the gravity of an ancient oracle, but your answers \
must be clear and correct.

Rules:
1. First decide whether the PASSAGES contain the answer to the QUESTION. Mentioning the same \
words is not enough: they must actually answer it.
2. If they do, answer using only information from the PASSAGES, in at most {n} sentence(s), \
each under 25 words. Do not add facts that are not in the PASSAGES.
3. Your answer will be spoken aloud: write plain sentences only. No lists, no markdown, no symbols. \
Say equations in words (for example "stress equals force divided by the original area").
4. Never mention passages, notes, documents or sources in the answer itself.

Reply in exactly this format:
ANSWERABLE: yes or no
SOURCES: the numbers of the passages you used, e.g. 1, 3 (or none)
HINT: if not answerable, the number of the ONE passage most related to the question, or none \
if none are related. Otherwise leave empty.
ANSWER: your answer (leave empty if not answerable)"""

# Spoken when the notes can't answer. One line each, so they sound natural aloud.
REFUSALS = [
    "The Head knows not. You never wrote of this, so I cannot speak of it.",
    "On this, the brass is silent. Nothing of it was ever written in your hand.",
    "I search your writings and find nothing of this. Ask me of what you have studied.",
]
# Used instead when the model names the nearest topic the notes do cover.
REFUSALS_WITH_HINT = [
    "The Head knows not. The nearest thing you have written of is {hint}.",
    "On this, the brass is silent. Your writings reach only as far as {hint}.",
    "I find nothing of this in your hand. Ask me instead of {hint}.",
]
NONSENSE = "The Head stirs, but finds nothing. Ask again, and ask of what you have studied."
REVEALS = [
    "This wisdom was yours all along: it came from your note on {notes}.",
    "I did but read your own words, from your note on {notes}.",
    "No oracle spoke here. You wrote this, in your note on {notes}.",
]


def sentences_for(tokens):
    n = tokens * config.SENTENCES_PER_TOKEN
    return max(config.MIN_SENTENCES, min(config.MAX_SENTENCES, n))


def join_names(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta kappa lambda mu nu xi pi rho "
         "sigma tau phi varphi chi psi omega Gamma Delta Theta Lambda Pi Sigma Phi Omega").split()
SAY = {r"\times": " times ", r"\cdot": " times ", r"\approx": " approximately ", r"\le": " at most ",
       r"\ge": " at least ", r"\propto": " is proportional to ", r"\infty": " infinity ",
       r"\sqrt": " root ", r"\Delta": " delta ", r"\%": " percent", "%": " percent", "=": " equals ",
       "≈": " approximately ", "×": " times ", "≤": " at most ", "≥": " at least "}


def to_speech(text):
    """Belt and braces: turn anything a text-to-speech engine would read literally into words."""
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r" \1 over \2 ", text)
    for k, v in SAY.items():
        text = text.replace(k, v)
    for g in GREEK:
        text = re.sub(rf"\\{g}\b", " " + g.replace("var", "") + " ", text)
    text = re.sub(r"(?<=\w)\s*/\s*(?=\w)", " over ", text)
    text = re.sub(r"\^\{?2\}?", " squared", text)
    text = re.sub(r"\^\{?3\}?", " cubed", text)
    text = re.sub(r"\^\{?([^{}\s]+)\}?", r" to the power \1", text)
    text = re.sub(r"_\{?([^{}\s]+)\}?", r" \1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)          # any LaTeX command left over
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"[*_#`>\[\]{}|]", "", text)        # markdown
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def parse(reply):
    def field(name):
        m = re.search(rf"^{name}:\s*(.*?)(?=^\w+:|\Z)", reply, re.S | re.M | re.I)
        return m.group(1).strip() if m else ""
    verdict = field("ANSWERABLE").lower()
    sources = [int(x) for x in re.findall(r"\d+", field("SOURCES"))]
    answer = field("ANSWER")
    if not verdict and not answer:          # model ignored the format: treat whole reply as the answer
        return True, [], "", reply.strip()
    return verdict.startswith("y"), sources, field("HINT"), answer


class Head:
    def __init__(self):
        self.index = Index()

    # -- model housekeeping -------------------------------------------------
    def warm_up(self):
        """Load both models onto the GPU so the first visitor doesn't wait."""
        t = time.time()
        embed(["warm up"], QUERY_PREFIX, keep_alive=config.KEEP_ALIVE)
        self._chat([{"role": "user", "content": "Say ready."}], max_tokens=5)
        return time.time() - t

    def unload(self):
        """Free the GPU memory (keep_alive 0 unloads immediately)."""
        for payload, endpoint in (({"model": config.CHAT_MODEL, "keep_alive": 0}, "generate"),
                                  ({"model": config.EMBED_MODEL, "input": "x", "keep_alive": 0}, "embed")):
            try:
                ollama(endpoint, payload, timeout=30)
            except OSError:
                pass

    def _chat(self, messages, max_tokens):
        r = ollama("chat", {
            "model": config.CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": config.KEEP_ALIVE,
            "options": {"temperature": config.TEMPERATURE, "num_ctx": 4096, "num_predict": max_tokens},
        })
        return r["message"]["content"], r

    # -- the ritual -----------------------------------------------------------
    def ask(self, question, tokens=None):
        if tokens is None:
            tokens = config.DEFAULT_TOKENS        # placeholder until the Bela sends a real count
        timings = {}
        t0 = time.time()
        hits = self.index.search(question, k=config.TOP_K, keep_alive=config.KEEP_ALIVE)
        timings["search_s"] = round(time.time() - t0, 3)
        n = sentences_for(tokens)
        result = {"question": question, "tokens": tokens, "sentences": n,
                  "passages": [{"note": h["note"], "section": h["section"],
                                "similarity": round(h["similarity"], 3)} for h in hits]}

        if not hits or max(h["similarity"] for h in hits) < config.SIMILARITY_FLOOR:
            result.update(answerable=False, answer=NONSENSE, reveal="", sources=[], timings=timings)
            return result

        passages = "\n\n".join(
            f"[{i}] ({h['note']} — {h['section'] or 'intro'})\n" + " ".join(h["text"].split()[: config.PASSAGE_MAX_WORDS])
            for i, h in enumerate(hits, 1))
        messages = [
            {"role": "system", "content": SYSTEM.format(n=n)},
            {"role": "user", "content": f"PASSAGES:\n{passages}\n\nQUESTION: {question}"},
        ]
        t1 = time.time()
        reply, raw = self._chat(messages, max_tokens=60 + 45 * n)
        timings["llm_s"] = round(time.time() - t1, 3)
        if raw.get("load_duration"):
            timings["model_load_s"] = round(raw["load_duration"] / 1e9, 3)
        timings["total_s"] = round(time.time() - t0, 3)

        answerable, sources, hint, answer = parse(reply)
        used = [hits[i - 1] for i in sources if 1 <= i <= len(hits)] or hits[:1]
        notes = list(dict.fromkeys(h["note"] for h in used))   # unique, in order

        if answerable and answer:
            result.update(answerable=True, answer=to_speech(answer),
                          reveal=random.choice(REVEALS).format(notes=join_names(notes)),
                          sources=[{"note": h["note"], "section": h["section"]} for h in used])
        else:
            # The hint is the TITLE of the passage the model picked (never model-written text),
            # and only if that passage is a reasonable match for the question.
            m = re.search(r"\d+", hint)
            near = hits[int(m.group()) - 1] if m and 1 <= int(m.group()) <= len(hits) else None
            if near and near["similarity"] >= config.HINT_MIN_SIMILARITY:
                title = near["note"]
                spoken = " ".join(w if w.isupper() else w.lower() for w in title.split())
                refusal = random.choice(REFUSALS_WITH_HINT).format(hint=spoken)
            else:
                refusal = random.choice(REFUSALS)
            result.update(answerable=False, answer=refusal, reveal="", sources=[])
        result["raw_reply"] = reply
        result["timings"] = timings
        return result
