# autodictate

Voice-activated, hands-free dictation for Linux (PipeWire).

Say **"hi babe"** → an audio cue plays → capture starts automatically. While
you speak, every pause of 2+ seconds produces a chunk that is transcribed and
typed into whatever app has focus, so text is always visible on screen.
**20 seconds** of continuous silence ends the session and the daemon returns to
idle. No keys, no manual start/stop.

Built on [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (CPU-only) and
the native ggml Silero VAD. The VAD gates everything — whisper only runs on
completed speech segments, which keeps CPU cost low.

## How it works

```
IDLE ──"hi babe" match──▶ CUE (play cue.wav) ──▶ CAPTURING
CAPTURING ──≥2 s silence──▶ transcribe + type chunk ──▶ CAPTURING
CAPTURING ──≥20 s continuous silence──▶ IDLE
```

- A rolling wake detector runs the VAD on idle audio; only a completed speech
  segment that fuzzy-matches the wake phrase (Levenshtein ≤ 2) starts a session.
  Ambient/TV speech cannot trigger it.
- The cue audio is **discarded** from capture, so "I'm listening." is never
  transcribed.
- Each pause-chunk is transcribed independently and typed immediately.
- A phrase spoken *during* a session never reaches the idle path, so it's
  ignored by construction.

## Requirements

System packages are listed in [`requirements.txt`](requirements.txt)
(pacman names; install the equivalents on other distros). In short:

- Linux with **PipeWire** + **WirePlumber** and a microphone.
- A C++17 toolchain: `g++`, `cmake`, `make`, `pkg-config`.
- `libpipewire-0.3`, `libevdev`, `libudev`.
- **whisper.cpp** (cloned and built separately — see below).
- The uinput interface (`/dev/uinput`) for typing (standard on Linux).

## Build

**1. Install dependencies** (Arch / CachyOS):
```bash
pacman -S base-devel pipewire wireplumber libevdev wget git
```

**2. Clone and build whisper.cpp** (v1.9.3, CPU-only):
```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
git checkout v1.9.3
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd ..
```
The default cmake build is CPU (AVX2), no Vulkan/CUDA — that's what we want.

**3. Download the models** into this repo's `models/` directory:
```bash
# from the whisper.cpp root:
cd whisper.cpp
./models/download-ggml-model.sh base.en        # STT model
./models/download-vad-model.sh silero-v6.2.0   # VAD model
cd ..
mkdir -p models
cp whisper.cpp/models/ggml-base.en.bin models/
cp whisper.cpp/models/ggml-silero-v6.2.0.bin models/
```
See [`models/README.md`](models/README.md) for sources.

**4. Build autodictate:**
```bash
./build.sh
```
This must compile with **zero warnings** (`-Wall -Wextra`). Output: `autodictate`.

## Configuration

Create `~/.config/voice-to-text/autodictate.conf` (plain `key=value`, `#`
comments, last duplicate wins, missing file → built-in defaults):

```ini
wake_phrase = hi babe
model       = ggml-base.en.bin
pause_chunk_s = 2
session_end_s = 20
```

| key | default | meaning |
|---|---|---|
| `wake_phrase` | `hi piya` | phrase that starts a session (fuzzy-matched) |
| `model` | — | whisper model filename under `models/` (or absolute path) |
| `pause_chunk_s` | `2` | silence ≥ this → transcribe + type a chunk |
| `session_end_s` | `20` | continuous silence ≥ this → end the session |

Precedence: CLI flags > `autodictate.conf` > built-in defaults. The daemon
prints a startup summary line with the effective values — read it first when
debugging.

## Usage

```bash
# daemon mode (default): idles for the wake phrase
./autodictate

# file mode: run the identical pause-loop over a 16 kHz mono S16 WAV
./autodictate --input some.wav

# test the wake path against a recording (prints match/no-match)
./autodictate --wake-test some.wav

# overrides
./autodictate --config /path/to/conf --model /path/to/model.bin
```

The daemon types into the focused app. It runs in the foreground; launch it
however you like (terminal, `setsid … &`, your app manager, …).

## Deep dive

For architecture, the code map (with line numbers), and the critical gotchas
(especially the **VAD 1280-sample frame requirement** and the **cue-discard
window**), read [`AUTODICTATE.md`](AUTODICTATE.md).

## License

[MIT](LICENSE)
