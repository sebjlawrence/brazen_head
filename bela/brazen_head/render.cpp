/*
 * Brazen Head - Bela side of the ritual (first working pipeline).
 *
 *   mic -> wait for speech -> record the question -> POST it to the laptop's /ritual
 *       -> receive the spoken answer + reveal (WAV) -> play it on the headphones -> listen again
 *
 * The laptop does speech-to-text, the LLM and the voice (python server.py).
 * HTTP for now; this will move to OSC (control) once that's covered in class.
 *
 * Wiring (as in Tutorial 2, step 1a; Bela Gem Stereo + capsule mic on a breadboard):
 *   mic - leg (joined to the case)  -> GND
 *   mic + leg -> LINE IN "L" pin (channel 0), and through a 2.2k resistor to 3.3V
 * Raise the input gain in the IDE's Audio tab (vertical-lines icon) if the levels printed below are tiny.
 */
#include <Bela.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ------------------------------------------------------------------ settings (edit these)
static const char*  SERVER          = "192.168.7.1"; // the laptop as seen from the Bela. Over USB this is
                                                     // usually 192.168.7.1 or 192.168.6.1 (ping it first)
static const int    PORT            = 8000;
static const int    TOKENS          = 2;      // placeholder until the break-beam counts real tokens
static const unsigned MIC_CHANNEL   = 0;      // audio input the mic is on
static const float  MIC_GAIN_DB     = 36.0f;  // analogue preamp gain for the mic, set at startup (0 to ~59.5 dB).
                                              // Aim for question peaks of 0.3-0.5 (the server saves them in heard/)
static const float  INPUT_BOOST     = 1.0f;   // extra DIGITAL gain (also boosts noise): only if MIC_GAIN_DB is maxed
static const float  OUTPUT_GAIN     = 0.5f;   // headphone volume (0..1)
static const float  START_FACTOR    = 4.0f;   // speech starts when the level is this many times the background
static const float  MIN_START_LEVEL = 0.005f; // ...and at least this loud (RMS, 0..1)
static const float  SILENCE_SECONDS = 1.2f;   // stop recording after this long quiet
static const float  MAX_SECONDS     = 8.0f;   // longest question
static const float  MIN_SECONDS     = 0.4f;   // shorter blips are ignored (a cough, a knock)
static const float  PREROLL_SECONDS = 0.3f;   // keep this much audio from before speech started
static const bool   THINKING_HUM    = true;   // slow brassy throb while the laptop thinks
static const float  THROB_PITCH_HZ  = 73.4f;  // fundamental (D2): lower = darker, more ominous
static const float  THROB_RATE_HZ   = 0.6f;   // swells per second (0.6 = one every ~1.7 s)
static const float  THROB_VOLUME    = 0.08f;  // loudness at the top of each swell
static const bool   PRINT_LEVELS    = true;   // print mic level twice a second (to set the thresholds)
static const float  MAX_ANSWER_SECONDS = 90.0f;

// ------------------------------------------------------------------ state
enum State { LISTENING, RECORDING, WAITING, PLAYING };
static std::atomic<int> gState(LISTENING);
static std::atomic<bool> gStop(false);
static std::thread gWorker;

static float gRate = 44100;
static std::vector<float> gPreroll;   static unsigned gPreWrite = 0;
static std::vector<float> gRec;       static unsigned gRecLen = 0;
static std::vector<float> gPlay;      static std::atomic<unsigned> gPlayLen(0); static unsigned gPlayPos = 0;

// level detection (20 ms windows)
static unsigned gWin = 882, gWinCount = 0; static float gWinSum = 0, gLevel = 0;
static float gBackground = 0.002f; static unsigned gSettle = 0;   // frames to ignore before triggering
static unsigned gQuietFrames = 0, gPrintCount = 0;
static float gDcX = 0, gDcY = 0;      // DC blocker
static float gThrobP1 = 0, gThrobP2 = 0, gThrobLfo = 0, gThrobGain = 0;   // thinking sound

// ------------------------------------------------------------------ WAV helpers (worker thread)
static void put16(FILE* f, uint16_t v) { fputc(v & 255, f); fputc(v >> 8, f); }
static void put32(FILE* f, uint32_t v) { put16(f, v & 0xffff); put16(f, v >> 16); }

static bool writeWav(const char* path, const float* x, unsigned n, unsigned rate)
{
	FILE* f = fopen(path, "wb");
	if(!f) return false;
	fwrite("RIFF", 1, 4, f); put32(f, 36 + n * 2); fwrite("WAVEfmt ", 1, 8, f);
	put32(f, 16); put16(f, 1); put16(f, 1); put32(f, rate); put32(f, rate * 2); put16(f, 2); put16(f, 16);
	fwrite("data", 1, 4, f); put32(f, n * 2);
	for(unsigned i = 0; i < n; i++) {
		float v = x[i] * INPUT_BOOST;
		v = v > 1 ? 1 : (v < -1 ? -1 : v);
		put16(f, (uint16_t)(int16_t)lrintf(v * 32767));
	}
	fclose(f);
	return true;
}

// Reads a 16-bit PCM WAV (as /ritual sends) into gPlay. Returns the number of frames (mixes to mono).
static unsigned readWav(const char* path)
{
	FILE* f = fopen(path, "rb");
	if(!f) return 0;
	std::vector<unsigned char> b;
	unsigned char buf[65536]; size_t r;
	while((r = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + r);
	fclose(f);
	if(b.size() < 12 || memcmp(&b[0], "RIFF", 4) || memcmp(&b[8], "WAVE", 4)) return 0;
	auto u16 = [&](size_t i) { return (unsigned)(b[i] | b[i + 1] << 8); };
	auto u32 = [&](size_t i) { return (unsigned)(b[i] | b[i + 1] << 8 | b[i + 2] << 16 | (unsigned)b[i + 3] << 24); };
	unsigned channels = 1, bits = 16, rate = 0;
	size_t p = 12;
	while(p + 8 <= b.size()) {
		unsigned size = u32(p + 4);
		if(!memcmp(&b[p], "fmt ", 4)) { channels = u16(p + 10); rate = u32(p + 12); bits = u16(p + 22); }
		else if(!memcmp(&b[p], "data", 4)) {
			if(bits != 16 || channels < 1) { printf("Answer WAV is %u-bit, expected 16\n", bits); return 0; }
			if(rate && fabsf((float)rate - gRate) > 1) printf("Warning: answer is %u Hz, Bela runs at %.0f Hz\n", rate, gRate);
			size_t end = std::min(b.size(), p + 8 + (size_t)size);
			unsigned frames = (unsigned)((end - p - 8) / (2 * channels));
			if(frames > gPlay.size()) frames = gPlay.size();
			for(unsigned i = 0; i < frames; i++) {
				float s = 0;
				for(unsigned c = 0; c < channels; c++)
					s += (int16_t)u16(p + 8 + (i * channels + c) * 2) / 32768.f;
				gPlay[i] = s / channels;
			}
			return frames;
		}
		p += 8 + size + (size & 1);
	}
	return 0;
}

static std::string headerValue(const char* path, const char* name)
{
	FILE* f = fopen(path, "r");
	if(!f) return "";
	char line[2048]; std::string out; size_t n = strlen(name);
	while(fgets(line, sizeof line, f))
		if(!strncasecmp(line, name, n) && line[n] == ':') {
			const char* v = line + n + 1; while(*v == ' ') v++;
			for(const char* c = v; *c && *c != '\r' && *c != '\n'; c++) {   // URL-decode
				if(*c == '%' && c[1] && c[2]) { char h[3] = {c[1], c[2], 0}; out += (char)strtol(h, nullptr, 16); c += 2; }
				else out += *c;
			}
		}
	fclose(f);
	return out;
}

// ------------------------------------------------------------------ worker: talks to the laptop
static void worker()
{
	while(!gStop) {
		if(gState != WAITING) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
		printf("Asking the laptop (%.1f s of audio)...\n", gRecLen / gRate);
		auto t0 = std::chrono::steady_clock::now();
		bool ok = writeWav("/tmp/bh_question.wav", gRec.data(), gRecLen, (unsigned)gRate);
		if(ok) {
			char cmd[512];
			snprintf(cmd, sizeof cmd,
				"curl -sS -f -m 120 -D /tmp/bh_headers.txt -o /tmp/bh_answer.wav "
				"-H 'Content-Type: audio/wav' --data-binary @/tmp/bh_question.wav "
				"'http://%s:%d/ritual?tokens=%d&rate=%d'", SERVER, PORT, TOKENS, (int)gRate);
			ok = system(cmd) == 0;
			if(!ok) printf("Request failed. Is server.py running, and can the Bela reach http://%s:%d/health ?\n", SERVER, PORT);
		}
		unsigned frames = ok ? readWav("/tmp/bh_answer.wav") : 0;
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		if(frames) {
			printf("Heard:  \"%s\"\n", headerValue("/tmp/bh_headers.txt", "X-Heard").c_str());
			printf("Notes:  %s\n", headerValue("/tmp/bh_headers.txt", "X-Notes").c_str());
			printf("Reply after %.2f s, %.1f s of speech. Speaking...\n", secs, frames / gRate);
			gPlayPos = 0;
			gPlayLen = frames;
			gState = PLAYING;
		} else {
			if(ok) printf("Got a reply, but it wasn't a WAV the Bela could read.\n");
			gState = LISTENING;
		}
	}
}

// ------------------------------------------------------------------ Bela
bool setup(BelaContext* context, void*)
{
	if(context->audioInChannels <= MIC_CHANNEL) { printf("No audio input %u\n", MIC_CHANNEL); return false; }
	gRate = context->audioSampleRate;
	gWin = (unsigned)(gRate * 0.02f);
	gPreroll.assign((size_t)(gRate * PREROLL_SECONDS), 0.f);
	gRec.assign((size_t)(gRate * (MAX_SECONDS + PREROLL_SECONDS)) + 1, 0.f);
	gPlay.assign((size_t)(gRate * MAX_ANSWER_SECONDS), 0.f);
	gSettle = (unsigned)gRate;   // learn the background level for 1 s before listening
	if(Bela_setAudioInputGain(MIC_CHANNEL, MIC_GAIN_DB))
		printf("Couldn't set the mic gain to %.1f dB: set it in the IDE's Audio tab instead\n", MIC_GAIN_DB);
	else
		printf("Mic gain on input %u set to %.1f dB\n", MIC_CHANNEL, MIC_GAIN_DB);
	gWorker = std::thread(worker);
	printf("Brazen Head: %.0f Hz, mic on input %u, server http://%s:%d. Speak after 1 s.\n",
	       gRate, MIC_CHANNEL, SERVER, PORT);
	return true;
}

static void startRecording()
{
	unsigned n = gPreroll.size();
	for(unsigned i = 0; i < n; i++) gRec[i] = gPreroll[(gPreWrite + i) % n];   // oldest first
	gRecLen = n;
	gQuietFrames = 0;
	gState = RECORDING;
	rt_printf("Listening to the question...\n");
}

static void endWindow()
{
	gLevel = sqrtf(gWinSum / gWin);
	gWinSum = 0; gWinCount = 0;
	int state = gState;
	float threshold = fmaxf(MIN_START_LEVEL, START_FACTOR * gBackground);
	if(state == LISTENING) {
		if(gSettle == 0 && gLevel > threshold) startRecording();
		else gBackground = 0.98f * gBackground + 0.02f * fminf(gLevel, 4 * gBackground + 1e-4f);
		if(PRINT_LEVELS && ++gPrintCount >= 25) {
			gPrintCount = 0;
			rt_printf("level %.4f  background %.4f  start above %.4f\n", gLevel, gBackground, threshold);
		}
	} else if(state == RECORDING) {
		if(gLevel < 0.5f * threshold) gQuietFrames += gWin; else gQuietFrames = 0;
		bool full = gRecLen >= gRec.size() - gWin;
		if(gQuietFrames >= SILENCE_SECONDS * gRate || full) {
			float speech = (gRecLen - gQuietFrames) / gRate - PREROLL_SECONDS;
			if(speech < MIN_SECONDS) { rt_printf("(too short, ignored)\n"); gState = LISTENING; }
			else gState = WAITING;   // the worker thread takes it from here
		}
	}
}

// Slow brassy throb: two slightly detuned brass-like tones (harmonics 1-8) that swell and fade.
// Louder = brighter, like a real brass instrument. Fades in/out over ~50 ms so it never clicks.
static float throb(bool on)
{
	const float twoPi = 2.f * (float)M_PI;
	gThrobGain += ((on ? 1.f : 0.f) - gThrobGain) * 0.0005f;
	if(gThrobGain < 1e-4f) { gThrobLfo = 0; return 0; }   // silent: restart from the bottom of a swell
	gThrobLfo += twoPi * THROB_RATE_HZ / gRate;  if(gThrobLfo > twoPi) gThrobLfo -= twoPi;
	gThrobP1  += twoPi * THROB_PITCH_HZ / gRate; if(gThrobP1 > twoPi) gThrobP1 -= twoPi;
	gThrobP2  += twoPi * THROB_PITCH_HZ * 1.004f / gRate; if(gThrobP2 > twoPi) gThrobP2 -= twoPi;
	float swell = 0.5f - 0.5f * cosf(gThrobLfo);
	swell *= swell;                                       // long quiet gap, rounded swell
	float rolloff = 0.35f + 0.45f * swell;                // more upper harmonics when louder
	float s = 0, w = 1;
	for(int k = 1; k <= 8; k++) {
		s += w * (sinf(k * gThrobP1) + sinf(k * gThrobP2));
		w *= rolloff;
	}
	return THROB_VOLUME * gThrobGain * (0.12f + 0.88f * swell) * s * 0.3f;
}

void render(BelaContext* context, void*)
{
	for(unsigned n = 0; n < context->audioFrames; n++) {
		// mic, with the DC offset removed (the bias voltage)
		float x = audioRead(context, n, MIC_CHANNEL);
		float y = x - gDcX + 0.995f * gDcY; gDcX = x; gDcY = y;

		int state = gState;
		if(state == LISTENING) {
			gPreroll[gPreWrite] = y; gPreWrite = (gPreWrite + 1) % gPreroll.size();
			if(gSettle) gSettle--;
		} else if(state == RECORDING && gRecLen < gRec.size()) {
			gRec[gRecLen++] = y;
		}
		if(state == LISTENING || state == RECORDING) {
			gWinSum += y * y;
			if(++gWinCount >= gWin) endWindow();
		}

		// output
		float out = 0;
		if(state == PLAYING) {
			if(gPlayPos < gPlayLen) out = gPlay[gPlayPos++] * OUTPUT_GAIN;
			else {
				gState = LISTENING; gSettle = (unsigned)(0.5f * gRate);   // ignore our own tail
				gWinSum = 0; gWinCount = 0;
				rt_printf("Done. Ask another question.\n");
			}
		}
		out += throb(state == WAITING && THINKING_HUM);
		for(unsigned ch = 0; ch < context->audioOutChannels; ch++) audioWrite(context, n, ch, out);
	}
}

void cleanup(BelaContext*, void*)
{
	gStop = true;
	if(gWorker.joinable()) gWorker.join();
}
