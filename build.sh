#!/usr/bin/env bash
# Build autodictate.
#
# Prereq: whisper.cpp cloned + built in ./whisper.cpp (see README "Build").
# Must compile with ZERO warnings (-Wall -Wextra).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d whisper.cpp/build/bin ]; then
  echo "error: whisper.cpp not found or not built in ./whisper.cpp/build/bin" >&2
  echo "       see README.md -> Build (steps 2-3)." >&2
  exit 1
fi

PKG_CONFIG_PATH=/usr/lib/pkgconfig g++ -std=c++17 -O2 -Wall -Wextra \
  autodictate.cpp \
  $(pkg-config --cflags --libs libpipewire-0.3) \
  $(pkg-config --cflags libevdev) -levdev -ludev \
  -I whisper.cpp/include -I whisper.cpp/ggml/include \
  -L whisper.cpp/build/bin -lwhisper -lggml -lggml-cpu \
  -Wl,-rpath,'$ORIGIN/whisper.cpp/build/bin' \
  -o autodictate

echo "built: $(pwd)/autodictate"
