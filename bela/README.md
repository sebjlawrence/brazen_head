# Bela side: the ritual pipeline

`brazen_head/render.cpp`: the Bela listens on its mic, records your question when you speak,
sends it to the laptop's `/ritual` endpoint, and plays the spoken answer and reveal on the headphones.

## Wiring (Tutorial 2, step 1a)

Breadboard to the Bela Gem Stereo's header pins (not the jack):
- mic **-** leg (the one joined to the metal case) -> **GND**
- mic **+** leg -> **LINE IN L** (channel 0)
- **2.2 kΩ** (red-red-black-brown-brown) from the mic + leg to **3.3V**

Check it first with the **passthrough** example (C++ > Fundamentals): tap the mic and you should hear it in one ear.

## Tonight's checklist

1. **Laptop: start the server**: `python server.py`. Wait for `Listening on ...`.
2. **Laptop: open the firewall for port 8000** (PowerShell **as Administrator**, once):
   `New-NetFirewallRule -DisplayName "Brazen Head 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
3. **Find the laptop's address as the Bela sees it.** Plug the Bela in over USB, open the Bela IDE
   (http://bela.local or http://192.168.7.2) and type these in its console at the bottom:
   `curl http://192.168.7.1:8000/health` (if that fails, try `192.168.6.1`).
   One of them should print `{"ok": true, ...}`. Put that address in `SERVER` at the top of render.cpp.
4. **Make a new project** in the IDE (e.g. `brazen_head`), and replace its `render.cpp` with this one
   (drag the file into the IDE's file list, or paste it in).
5. **Mic gain**: set by `MIC_GAIN_DB` at the top of render.cpp (36 dB to start). The Audio tab still changes it live while testing; copy the value that works back into `MIC_GAIN_DB`.
   Run the project and watch the `level ... background ...` lines: speaking should push `level`
   well above `start above`. Raise the gain (or `INPUT_BOOST`) if it doesn't; raise `START_FACTOR` if
   it triggers on its own.
6. **Ask a question.** The console shows what Whisper heard, which notes it used and how long it took.
   You'll hear a quiet low hum while the laptop thinks, then the answer.

## If something's wrong

| Symptom | Check |
|---|---|
| `Request failed` | Server running? Step 3's `curl .../health` works from the Bela? Firewall rule added? |
| It never starts recording | Level too low: 2.2k to 3.3V present? mic polarity? resistor leg touching the mic case? gain; `MIN_START_LEVEL` |
| It records forever (8 s every time) | Background noise too close to speech: raise `START_FACTOR` or lower the gain |
| Whisper hears nothing / nonsense | `INPUT_BOOST` up; check `ritual_log.jsonl` on the laptop; the question WAV is at `/tmp/bh_question.wav` on the Bela (copy it off and listen) |
| Answer is too loud/quiet | `OUTPUT_GAIN` |

Settings are all at the top of render.cpp. `TOKENS` is a placeholder until the break-beam counts tokens.
Every exchange is logged on the laptop in `ritual_log.jsonl` (what it heard, the answer, the notes, timings).
