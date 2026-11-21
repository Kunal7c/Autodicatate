# Models

autodictate needs two model files in this directory (next to the `autodictate`
binary). They are **not** committed (too large) — download them once.

| file | what | size | source |
|---|---|---|---|
| `ggml-base.en.bin` | whisper STT model (English) | ~148 MB | standard whisper.cpp model |
| `ggml-silero-v6.2.0.bin` | ggml Silero VAD | ~0.9 MB | ggml-org/whisper-vad |

## Download

From the **whisper.cpp** root directory (after cloning it — see README "Build"):

```bash
cd whisper.cpp
./models/download-ggml-model.sh base.en        # -> models/ggml-base.en.bin
./models/download-vad-model.sh silero-v6.2.0   # -> models/ggml-silero-v6.2.0.bin
cd ..

# copy them into this repo's models/ directory:
mkdir -p models
cp whisper.cpp/models/ggml-base.en.bin models/
cp whisper.cpp/models/ggml-silero-v6.2.0.bin models/
```

## Direct download URLs (if you prefer)

- `ggml-base.en.bin` — https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
- `ggml-silero-v6.2.0.bin` — https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin

Other whisper models (`tiny`, `small`, …) work too if you set `model =` in
`~/.config/voice-to-text/autodictate.conf`. `base.en` is the tested default.
