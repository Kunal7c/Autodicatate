# Auto Dictate — Manual

Voice-activated, hands-free dictation daemon. Say **"hi babe"** → an audio
cue plays ("Hi Kunal, I'm listening.") → capture starts automatically → every
2 s pause produces a transcribed + typed chunk → **20 s** of continuous silence
ends the session and the daemon returns to idle. No keys, no manual start/stop.

> **This manual is the authoritative reference for the program.** The live
> wake phrase is **`hi babe`** (set in the config file; the built-in default
> is `hi piya`), the session-end is **20 s**, and the **cue audio is
> discarded**. Where any other doc disagrees, trust this manual.

Single source file: **`autodictate.cpp`** (~2,800 lines). Build output:
`autodictate` (same dir). Everything below points into that one file.

---

## 1. Architecture / data flow

One **persistent** PipeWire capture stream runs for the daemon's whole life
(created once at startup, reused across every session — see §6 Gotcha 3). Its
`OnStreamProcess` callback routes each incoming buffer by state:

```
                    OnStreamProcess (per buffer)
                            │
        ┌───────────────────┼───────────────────────────────┐
   idle (no session)    session_active,            session_active,
   feed idle_capture    capturing=true             capturing=false
        │                append to app->pcm        (chunk transcribing:
        │                (unless in cue-discard    audio is DROPPED)
        │                 window, see §6 Gotcha 2)
        ▼
  main loop idle branch:
    accumulate idle_capture to whole 1280-sample frames,
    feed WakeDetector::Feed → Silero VAD gate
      └─ completed speech segment → whisper → fuzzy match "hi babe"
            └─ MATCH → StartSession (play cue, open capture, set discard window)

  main loop session branch:
    VADFeed(app->pcm) → Silero VAD
      └─ pause ≥ pause_chunk_s (2 s) → ProcessChunk: transcribe + type, resume
      └─ silence ≥ session_end_s (20 s) → EndSession → TranscribeAndType → IDLE
```

**The VAD is the single gate.** whisper.cpp (STT) only ever runs on a
*completed* VAD speech segment — never on continuous audio. This is what keeps
CPU cost low and what makes the whole thing work on a CPU-only box. Do not
bypass the VAD to "just always transcribe."

Typing is **uinput** (a synthetic keyboard), US layout, into whatever app has
focus. Text is trimmed, keeps punctuation, no trailing newline.

### State machine (current values)

```
IDLE ──"hi babe" match──▶ CUE (play cue.wav, open capture) ──▶ CAPTURING
CAPTURING ──≥2 s silence──▶ transcribe + type chunk ──▶ CAPTURING
CAPTURING ──≥20 s continuous silence──▶ IDLE
```

A phrase spoken *during* an active session never reaches the idle branch, so it
is ignored by construction.

---

## 2. Code map (where things are)

Line numbers refer to the current `autodictate.cpp`.

| Symbol | Line | What it does |
|---|---|---|
| `kRequestRate` | 114 | 16000 Hz capture rate (hard constant) |
| `kVadFrameSamples` | 119 | **1280** — Silero VAD frame size (80 ms @ 16 kHz). Central to Gotcha 1 |
| `kDefaultPauseChunkSeconds` | 125 | 2.0 s default chunk pause |
| `kDefaultSessionEndSeconds` | 126 | **20.0 s** default session end |
| `kCueDiscardMarginSeconds` | 130 | 0.5 s margin added to cue length for the discard window |
| `kDefaultWakePhrase` | 141 | `"hi piya"` built-in default (live value is `hi babe` via config) |
| `kWakeRingSamples` | 143 | 8 s rolling wake window |
| `struct App` | 427 | All daemon state (stream, pcm, VAD, probe, `discard_until_t`, …) |
| `struct WakeDetector` | 406 | Idle wake-phrase detector (VAD-gated) |
| `VADDetectSpeech` | 560 | Run one Silero VAD frame, return speech probability |
| `VADFeed` | 611 | Feed pcm through the VAD; drives chunk/stop decisions |
| `OnStreamProcess` | 757 | PipeWire buffer callback — **routes idle vs session audio** |
| `SetupCaptureStream` | 1449 | Build the persistent capture stream on the default input |
| `ReadWavDurationSeconds` | 1502 | Parse a WAV header → duration (used for the cue discard window) |
| `PlayCue` | 1548 | Play `cue.wav` via detached `paplay` (never blocks) |
| `StartSession` | 1578 | Begin a session: reset state, play cue, **set `discard_until_t`** |
| `EndSession` | 1627 | End the session, flush trailing VAD frame, run tail |
| `TranscribeAndType` | 2290 | Whisper + uinput typing for the whole session |
| `ProcessChunk` | 2333 | Whisper + uinput typing for one pause-chunk |
| `WakeDetector::Feed` | 2178 | **The function with the 1280-sample requirement (Gotcha 1)** |
| main loop | 2428 | `while(true)` — `pw_loop_iterate(0)` + idle/session branches |
| `main` | 2576 | Arg parse, model load, PipeWire setup, enter main loop |

---

## 3. Build

```bash
cd /home/kunalc/SATA1TB/AI/Pi/SpeachToText
PKG_CONFIG_PATH=/usr/lib/pkgconfig g++ -std=c++17 -O2 -Wall -Wextra \
  autodictate.cpp \
  $(pkg-config --cflags --libs libpipewire-0.3) \
  $(pkg-config --cflags libevdev) -levdev -ludev \
  -I whisper.cpp/include -I whisper.cpp/ggml/include \
  -L whisper.cpp/build/bin -lwhisper -lggml -lggml-cpu \
  -Wl,-rpath,'$ORIGIN/whisper.cpp/build/bin' -o autodictate
```

**Requirement: zero warnings** (`-Wall -Wextra`). The build links against the
local `whisper.cpp` build (`whisper.cpp/build/bin`) via rpath — keep that tree
in place.

Models (in `models/`): `ggml-base.en.bin` (whisper STT, in use),
`ggml-silero-v6.2.0.bin` (ggml Silero VAD, in use). `silero_vad.onnx` and the
`ggml-{tiny,small}.en.bin` models are present but **not** used by this build.

---

## 4. Configuration

Two plain `key=value` files (`#` comments, blank lines ignored, last duplicate
key wins, missing file = defaults, unknown keys = warning + ignored):

**`~/.config/voice-to-text/autodictate.conf`** (path overridable with
`--config PATH`) — wake phrase + pause-loop thresholds:

| key | default | notes |
|---|---|---|
| `wake_phrase` | `hi piya` | live value is **`hi babe`** (set in the file) |
| `pause_chunk_s` | `2` | silence ≥ this → transcribe + type a chunk |
| `session_end_s` | `20` | continuous silence ≥ this → end session |
| `model` | (dictate.conf) | filename under `models/` or absolute path |

**`~/.config/voice-to-text/dictate.conf`** (fixed path, always read) — legacy
VAD/model config. The autodictate build reads `model` and `vad_threshold` from
it; the old `start_key`/`stop_key`/`silence_seconds`/`seconds` keys are
**ignored** (they print "unknown key" warnings at startup — harmless).

**Precedence:** CLI flags > `autodictate.conf` > `dictate.conf` > built-in
defaults. Every mode prints a startup summary line with the effective values —
**read that line first when debugging config.**

Current live config:
```
autodictate.conf:  wake_phrase = hi babe
dictate.conf:      model = ggml-base.en.bin
                   vad_threshold = 0.5
```
So the effective session-end is the **20 s default** (nothing overrides it).

---

## 5. CLI

```
./autodictate [--input FILE] [--config PATH] [--vad-model PATH]
              [--model PATH] [--wake-test FILE]
```

- **no flags** → daemon mode (default). Idles for the wake phrase.
- `--input FILE` → **file mode**: reads a 16 kHz mono S16 WAV and runs the
  *identical* pause-loop over it with no real-time pacing. Great for testing
  chunking/session-end without a mic.
- `--wake-test FILE` → feed a WAV through the **idle wake path** and print the
  match/no-match decision. Use this to validate the wake phrase against a
  recording.
- `--config PATH`, `--model PATH`, `--vad-model PATH` → overrides.

---

## 6. Critical design decisions & gotchas

These are the things that will bite you if you change the code. Read them before
touching the capture path.

### Gotcha 1 — the VAD needs whole 1280-sample frames (the big one)
The live PipeWire source emits **~341-sample buffers (21 ms)**. But
`WakeDetector::Feed` (line 2178) only runs a Silero VAD frame when a *single*
`Feed()` call receives **≥ `kVadFrameSamples` (1280)** samples — its inner loop
is `while (pos + 1280 <= n)`. Feeding a 341-sample chunk directly runs **zero**
frames, so the VAD silently never ran on live audio and the wake phrase never
triggered. File mode worked because it feeds the whole file in one call.

**The fix (in the main-loop idle branch, ~line 2540):** accumulate in
`idle_capture` until it holds a full 1280-sample frame, then feed only whole
frames (`to_feed = (size / 1280) * 1280`) and `erase` the remainder for next
time. **Do not "simplify" this back to feeding whatever is in `idle_capture`
each tick** — that re-introduces the starvation. Symptom if regressed: stream
reports STREAMING, `OnStreamProcess` fires ~47×/s, but no VAD frames ever run
and wake never triggers.

### Gotcha 2 — the cue audio must be discarded
The mic picks up the cue ("Hi Kunal, I'm listening.") while it plays, and it
would otherwise be transcribed as dictation. `StartSession` (line 1578) sets:
```
discard_until_t = now + (cue_duration + kCueDiscardMarginSeconds /* 0.5 s */)
```
where `cue_duration` comes from `ReadWavDurationSeconds(cue.wav)` (line 1502;
the cue is 2.28 s). `OnStreamProcess` drops samples while
`now < discard_until_t`. Because the session VAD is fed from `app->pcm`
(main-loop session branch), dropping from `pcm` keeps the cue out of **both**
the VAD and the transcript. A missing/unreadable `cue.wav` falls back to a 2.5 s
window.

### Gotcha 3 — the stream is persistent; do not recreate per session
One capture stream for the daemon's life. Per-session stream creation was
removed (it ate the first words and under-delivered ~21% of nominal). The main
loop calls `pw_loop_iterate(app->loop, 0)` **non-blocking** every iteration and
never blocks on the loop. A dead stream is destroyed and recreated with backoff
(the in-flight session is aborted).

### Gotcha 4 — VAD gates everything
whisper only runs on a completed VAD segment. This is the CPU-cost control and
the reason ambient/TV speech doesn't trigger capture. Don't remove the gate.

### Gotcha 5 — capture is off while a chunk transcribes
During `ProcessChunk` (≈1–2 s on CPU) `capturing=false` and incoming audio is
**dropped** (accepted constraint, in the idle-branch `else if (app->session_active)`).
The 2 s / 20 s silence timeline is measured across chunks and is **not** reset
per chunk.

### Gotcha 6 — peak normalization
Captured audio is peak-normalized to ~0.95 before whisper (the quiet-mic fix
from 2026-09-01). A pure-silence chunk (max magnitude 0) prints a warning and
types nothing.

### Gotcha 7 — whisper sentinels are skipped
`[BLANK_AUDIO]`, `[NON-ENGLISH SPEECH]`, `[PHONETICS]` segments produce no typed
output.

---

## 7. Debugging

**First step, always:** read the daemon's startup summary line — it prints the
effective `wake_phrase`, `pause_chunk_s`, `session_end_s`, model, and config
path. Most "it's not doing X" bugs are a config/value mismatch.

**File mode for repro (no mic needed):**
```bash
./autodictate --input some_16k_mono.wav          # chunking / session-end
./autodictate --wake-test some_16k_mono.wav      # wake-path decision only
```
Any bug that reproduces in file mode is *not* in the live-stream path — it's in
VAD/whisper/matching. A bug that only happens live is in stream routing (see
Gotchas 1 & 3).

**Live debug logging pattern:** the codebase uses short, per-second,
gated-by-state log lines (e.g. `[dbg-vad]`, `[dbg-cb]`, `[dbg-branch]`) that
print at most once per wall-second. Add the same way when diagnosing, and
**remove all `TEMP-DEBUG` blocks before a final build** (the build must stay
zero-warning). Run the daemon to a log:
```bash
pkill -x autodictate; sleep 1
setsid ./autodictate > _dbg.log 2>&1 < /dev/null & disown
```

**Key invariants to check when the live path misbehaves:**
- `OnStreamProcess` firing ~47×/s and dequeuing ~16000 samples/s → audio is
  arriving (stream is healthy).
- `VADFeed`/`VADDetectSpeech` actually being called on live audio → Gotcha 1 is
  intact.
- `session_active` / `capturing` values match the expected branch.

---

## 8. Files in this repo

| File | Status |
|---|---|
| `autodictate.cpp` | **The** source. Edit this for any change. |
| `autodictate` | Build output (not committed). Rebuild with `build.sh`. |
| `cue.wav` | The "I'm listening." cue. Its duration drives the discard window. |
| `models/` | whisper + Silero VAD models (download separately — `models/README.md`). |
| `build.sh` | The build command, wrapped. |
| `README.md` | Quick start (build, config, usage). |
| `requirements.txt` | System dependencies. |
| `LICENSE` | MIT. |

---

## 9. Known limitations (accepted)

- **Chunk seams:** each chunk is transcribed independently, so a seam can add a
  stray period ("I want to. Order pizza."). Cosmetic, accepted.
- **No capture during a chunk** (≈1–2 s on CPU) — audio in that window is dropped.
- **Quiet mic:** VAD sensitivity (`vad_threshold`, default 0.5) may need live
  tuning for a given mic. It worked as-is on Kunal's Fifine mic.

---

## 10. Related

- `README.md` — quick start, build, config, usage.
- `build.sh` — the build command, wrapped.
- `requirements.txt` — system dependencies.
- `models/README.md` — where to get the model files.
