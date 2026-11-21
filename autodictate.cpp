// autodictate.cpp — PipeWire capture, VAD pause loop, uinput typing,
// dictation daemon, configuration file (D12), streaming per-chunk typing.
//
// Stripped down from dictate-copy.cpp (autodictate ticket 01): the
// standalone key-watch (--keys) and typing (--type) diagnostic modes are
// removed. Both modes implement the pause-loop state machine (tickets
// 02/03):
//
//   IDLE ──"hi piya"──▶ CAPTURING
//   CAPTURING ──≥2 s silence──▶ transcribe + type chunk ──▶ CAPTURING
//   CAPTURING ──≥20 s continuous silence──▶ IDLE
//
// Dictation daemon mode (default):
//   Loads the Silero VAD and whisper STT models (CPU-only) once, then idles
//   listening for the wake phrase "hi piya" (ticket 02). When a completed
//   speech segment fuzzy-matches the phrase, a capture session of the
//   system default input (the source WirePlumber marks as "default") at
//   16 kHz mono opens, feeding every frame through the whisper.cpp native
//   ggml Silero VAD. Inside a session:
//     - a pause of pause_chunk_s (default 2 s, autodictate.conf) with
//       buffered speech produces a chunk: capture stops, the buffer is
//       transcribed (whisper.cpp greedy
//       sampling) and streamed per-chunk (T-2.7) so each completed chunk is
//       typed immediately, the buffer is cleared, and capture resumes
//       immediately. This loops for the whole session — there is no hard
//       time cap.
//     - session_end_s (default 20 s, autodictate.conf) of continuous
//       silence ends the session and the daemon returns to idle.
//   While a chunk is transcribing, capture is off and incoming audio is
//   dropped (accepted design constraint). The typed text keeps punctuation,
//   trims leading/trailing whitespace (no trailing newline). Each chunk
//   prints "dictate: typed: ..." and a one-line chunk summary; a session
//   end prints a one-line summary (stop reason, chunks typed, total audio).
//   A per-session failure is non-fatal: the error is printed and the daemon
//   returns to idle.
//
// File mode:
//   --input FILE reads a 16 kHz mono S16 WAV and runs the identical
//   pause-loop session over it with no real-time pacing (one VAD frame at a
//   time): 2 s pauses produce typed chunks, 20 s of continuous silence ends
//   the session, and any audio still buffered at input end is flushed as a
//   final chunk. The run then exits.
//
// Configuration (D12):
//   Two config files, both plain key=value ('#' starts a comment, blank
//   lines are ignored, the last duplicate key wins; a missing file is not
//   an error — built-in defaults apply; unknown keys print a warning and
//   are ignored; an invalid value prints a diagnostic and exits 2):
//   ~/.config/voice-to-text/dictate.conf — the legacy hotkey/VAD config,
//   always read at its fixed path. Keys: hold_key (a '+'-separated list of
//   one or more of ctrl/shift/alt/super, case-insensitive, e.g.
//   "ctrl+alt"; the old start_key/stop_key keys are gone and hit the
//   unknown-key warning), model (a filename under <exedir>/models/ or an
//   absolute path), vad_threshold (0..1), gpu_device (reserved: any value
//   is accepted and ignored with a startup note — this build is
//   CPU-only).
//   ~/.config/voice-to-text/autodictate.conf — the wake phrase and
//   pause-loop thresholds (ticket 04); --config PATH overrides its path
//   (dictate.conf keeps its fixed path). Keys: wake_phrase (default
//   "hi piya"), pause_chunk_s (2), session_end_s (20), model (a filename
//   under <exedir>/models/ or an absolute path). Precedence: CLI flags >
//   autodictate.conf > dictate.conf > built-in defaults. Every mode prints
//   a startup summary line with the effective values.
//
// Default-input discovery uses the cleaner PipeWire mechanism rather than
// guessing property names: WirePlumber publishes a "default" metadata object
// (registry global of type PipeWire:Interface:Metadata with metadata.name =
// "default") whose property "default.audio.source" names the current default
// input node. Fallback: first node with media.class == "Audio/Source"
// (preferring physical alsa_input nodes).
//
// Build:  PKG_CONFIG_PATH=/usr/lib/pkgconfig g++ -std=c++17 -O2 -Wall -Wextra
//           autodictate.cpp $(pkg-config --cflags --libs libpipewire-0.3)
//           $(pkg-config --cflags libevdev) -levdev -ludev
//           -I whisper.cpp/include -I whisper.cpp/ggml/include
//           -L whisper.cpp/build/bin
//           -lwhisper -lggml -lggml-cpu
//           -Wl,-rpath,'$ORIGIN/whisper.cpp/build/bin' -o autodictate
// Usage:  ./autodictate [--input FILE] [--config PATH]
//           [--vad-model PATH] [--model PATH] [--wake-test FILE]

#include <pipewire/pipewire.h>
#include <pipewire/extensions/metadata.h>
#include <spa/param/audio/raw-utils.h>
#include <spa/pod/builder.h>
#include <whisper.h>
#include <libevdev/libevdev.h>  // evdev hotkey observation (T-2.3)
#include <linux/input.h>  // EV_KEY/EV_SYN and KEY_* scan codes (T-2.4)
#include <linux/uinput.h>  // temporary uinput keyboard (T-2.4)

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>    // scan /dev/input
#include <fcntl.h>     // open(O_CLOEXEC)
#include <poll.h>      // watch loop
#include <sys/ioctl.h> // UI_DEV_CREATE and friends (T-2.4)
#include <sys/stat.h>  // stat (cue.wav presence, ticket 04)
#include <unistd.h>    // readlink, close, usleep, fork, _exit

namespace {

constexpr uint32_t kRequestRate = 16000;
constexpr uint32_t kRequestChannels = 1;
constexpr const char *kWavPath = "capture.wav";
constexpr double kMetadataTimeout = 1.0;  // s, for the "default" metadata global
constexpr double kSyncTimeout = 5.0;      // s, for the registry sync barrier
constexpr int kVadFrameSamples = 1280;  // Silero VAD frame: 80 ms at 16 kHz
// ---------------------------------------------------------------------------
// Pause loop (ticket 03/04) — built-in defaults only. The runtime values
// are App::pause_chunk_samples / App::session_end_samples, set from
// ~/.config/voice-to-text/autodictate.conf in main.
// ---------------------------------------------------------------------------
constexpr double kDefaultPauseChunkSeconds = 2.0;   // silence >= this: chunk (transcribe+type, resume)
constexpr double kDefaultSessionEndSeconds = 20.0;  // continuous silence >= this: session ends
// Extra headroom discarded after the session-start cue, on top of the cue's
// own duration: covers paplay startup latency + any speech tail, so the cue
// audio ("Hi Kunal, I'm listening.") is never transcribed as dictation.
constexpr double kCueDiscardMarginSeconds = 0.5;
// Minimum chunk audio for transcription: whisper hallucinates confidently
// on very short input (a short tail becomes a wrong sentence), so chunks
// with less audio are reported but not typed.
constexpr double kMinAudioSeconds = 1.0;
constexpr const char *kDefaultVadModelName = "ggml-silero-v6.2.0.bin";

// ---------------------------------------------------------------------------
// Wake phrase (ticket 02/04) — built-in default; the runtime value is
// App::wake_phrase, set from autodictate.conf in main.
// ---------------------------------------------------------------------------
constexpr const char *kDefaultWakePhrase = "hi piya";
constexpr int kWakeMaxEdits = 2;  // Levenshtein tolerance (per word / whole)
constexpr int64_t kWakeRingSamples = kRequestRate * 8;  // 8 s rolling window
constexpr int64_t kWakeMaxSilenceSamples = kRequestRate / 2;  // 0.5 s
constexpr double kWakeMinSeconds = 0.4;  // below: too short for whisper
constexpr double kWakeMaxSeconds = 6.0;  // above: ordinary speech, skip

// whisper.cpp (v1.9.3) routes all whisper/ggml log output through one global
// callback registered with whisper_log_set(); the VAD emits a per-frame INFO
// line per fed frame (~12 per second of audio). Register a quiet sink: keep
// warnings and errors visible on stderr, suppress the INFO chatter so the
// daemon stays quiet.
void WhisperLogSink(ggml_log_level level, const char *text, void *user_data) {
  (void)user_data;
  if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// WirePlumber publishes "default.audio.source" as a JSON object string:
//   {"name":"alsa_input.pci-0000_12_00.6.analog-stereo"}
// Older or non-WirePlumber setups may publish a plain node name. Extract the
// name from either form; return "" when there is no usable name (e.g. null).
std::string ExtractSourceName(const char *value) {
  if (!value || std::strcmp(value, "null") == 0) return {};
  const std::string v(value);
  const std::size_t pos = v.find("\"name\"");
  if (pos != std::string::npos) {
    const std::size_t colon = v.find(':', pos);
    if (colon != std::string::npos) {
      const std::size_t q1 = v.find('"', colon + 1);
      const std::size_t q2 =
          (q1 != std::string::npos) ? v.find('"', q1 + 1) : std::string::npos;
      if (q1 != std::string::npos && q2 != std::string::npos)
        return v.substr(q1 + 1, q2 - q1 - 1);
    }
    return {};  // e.g. {"name":null}
  }
  return v;  // plain-string form
}

// Directory containing the executable (via /proc/self/exe), so the default
// VAD model path resolves relative to the binary, not the current working
// directory.
std::string ExecutableDirectory() {
  char buf[4096];
  const ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0) return {};
  buf[len] = '\0';
  const std::string path(buf);
  const std::size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

// ---------------------------------------------------------------------------
// Configuration (T-2.6, D12): ~/.config/voice-to-text/dictate.conf
// ---------------------------------------------------------------------------

constexpr int kModCtrl = 1 << 0;
constexpr int kModShift = 1 << 1;
constexpr int kModAlt = 1 << 2;
constexpr int kModSuper = 1 << 3;

std::string ToLower(std::string s) {
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string Trim(const std::string s) {
  const std::string ws = " \t\r\n";
  const std::size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return {};
  const std::size_t b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}

// Parse a push-to-talk chord: a '+'-separated list of one or more of
// ctrl/shift/alt/super (case-insensitive, e.g. "ctrl+alt" or "alt+ctrl").
// *mods is the resulting bitmask; *label is the normalized spelling in
// canonical order (ctrl,shift,alt,super) used in summaries and diagnostics.
// False on any other input (empty, an unknown token, or no modifiers).
bool ParseHoldCombo(const std::string &text, int *mods, std::string *label) {
  const std::string t = ToLower(Trim(text));
  if (t.empty()) return false;
  int m = 0;
  std::size_t start = 0;
  while (start != std::string::npos) {
    const std::size_t plus = t.find('+', start);
    const std::string token =
        Trim(t.substr(start, plus == std::string::npos
                          ? std::string::npos
                          : plus - start));
    const int bit =
        token == "ctrl" ? kModCtrl
        : token == "shift" ? kModShift
        : token == "alt" ? kModAlt
        : token == "super" ? kModSuper
        : 0;
    if (bit == 0) return false;
    m |= bit;
    start = (plus == std::string::npos) ? std::string::npos : plus + 1;
  }
  if (m == 0) return false;
  std::string canon;
  const int order[4] = {kModCtrl, kModShift, kModAlt, kModSuper};
  const char *names[4] = {"ctrl", "shift", "alt", "super"};
  for (int i = 0; i < 4; ++i) {
    if ((m & order[i]) == 0) continue;
    if (!canon.empty()) canon += "+";
    canon += names[i];
  }
  *mods = m;
  *label = canon;
  return true;
}

// Values from the config file. Unset: the hold combo has hold_mods == 0, the
// numeric fields hold sentinel -1, and the strings are empty.
struct Config {
  int hold_mods = 0;  // bitmask: kModCtrl | kModShift | kModAlt | kModSuper
  std::string hold_label;
  std::string model;
  double vad_threshold = -1.0;
  std::string gpu_device;
};

// Load a config file (D12): plain key=value lines, '#' comments, blank lines
// ignored, last duplicate key wins. Unknown keys print a warning and are
// ignored. A missing file is not an error (built-in defaults apply). Returns
// false with a diagnostic in *error only for invalid values (the caller
// exits 2).
bool LoadConfig(const std::string &path, Config *config, std::string *error) {
  std::ifstream in(path);
  if (!in) {
    if (errno == ENOENT) return true;  // missing: not an error (D12)
    std::fprintf(stderr,
                 "dictate: config: warning: cannot read %s: %s — using "
                 "defaults\n",
                 path.c_str(), std::strerror(errno));
    return true;
  }
  std::string line;
  int line_no = 0;
  auto fail = [&](const std::string &why) {
    *error = "dictate: config: " + path + ":" + std::to_string(line_no) +
             ": " + why;
    return false;
  };
  while (std::getline(in, line)) {
    line_no++;
    const std::string t = Trim(line);
    if (t.empty() || t[0] == '#') continue;
    const std::size_t eq = t.find('=');
    if (eq == std::string::npos) return fail("expected key=value");
    const std::string key = Trim(t.substr(0, eq));
    const std::string value = Trim(t.substr(eq + 1));
    if (key == "hold_key") {
      if (!ParseHoldCombo(value, &config->hold_mods, &config->hold_label))
        return fail("invalid hold_key value '" + value + "'");
    } else if (key == "model") {
      if (value.empty()) return fail("empty model value");
      config->model = value;
    } else if (key == "vad_threshold") {
      char *end = nullptr;
      const double v = std::strtod(value.c_str(), &end);
      if (end == value.c_str() || *end != '\0' || v < 0.0 || v > 1.0)
        return fail("invalid vad_threshold value '" + value + "'");
      config->vad_threshold = v;
    } else if (key == "gpu_device") {
      config->gpu_device = value;  // reserved: any value is ignored (D12)
    } else {
      std::fprintf(stderr,
                   "dictate: config: warning: ignoring unknown key '%s'\n",
                   key.c_str());
    }
  }
  return true;
}

// Values from the autodictate config file (ticket 04):
// ~/.config/voice-to-text/autodictate.conf (override with --config PATH).
// Unset: the strings are empty and the numeric fields hold sentinel -1.0.
struct AutoConfig {
  std::string wake_phrase;
  double pause_chunk_s = -1.0;
  double session_end_s = -1.0;
  std::string model;
};

// Load the autodictate config file (ticket 04): the same plain key=value
// syntax as LoadConfig — '#' comments, blank lines ignored, last duplicate
// key wins. Unknown keys print a warning and are ignored. A missing file is
// not an error (built-in defaults apply). Returns false with a diagnostic in
// *error only for invalid values (the caller exits 2).
bool LoadAutoConfig(const std::string &path, AutoConfig *config,
                    std::string *error) {
  std::ifstream in(path);
  if (!in) {
    if (errno == ENOENT) return true;  // missing: not an error (ticket 04)
    std::fprintf(stderr,
                 "dictate: config: warning: cannot read %s: %s — using "
                 "defaults\n",
                 path.c_str(), std::strerror(errno));
    return true;
  }
  std::string line;
  int line_no = 0;
  auto fail = [&](const std::string &why) {
    *error = "dictate: config: " + path + ":" + std::to_string(line_no) +
             ": " + why;
    return false;
  };
  // A threshold is a strictly-positive number of seconds.
  auto parse_seconds = [&](const std::string &value) {
    char *end = nullptr;
    const double v = std::strtod(value.c_str(), &end);
    return end != value.c_str() && *end == '\0' && v > 0.0;
  };
  while (std::getline(in, line)) {
    line_no++;
    const std::string t = Trim(line);
    if (t.empty() || t[0] == '#') continue;
    const std::size_t eq = t.find('=');
    if (eq == std::string::npos) return fail("expected key=value");
    const std::string key = Trim(t.substr(0, eq));
    const std::string value = Trim(t.substr(eq + 1));
    if (key == "wake_phrase") {
      if (value.empty()) return fail("empty wake_phrase value");
      config->wake_phrase = value;
    } else if (key == "pause_chunk_s") {
      if (!parse_seconds(value))
        return fail("invalid pause_chunk_s value '" + value + "'");
      config->pause_chunk_s = std::strtod(value.c_str(), nullptr);
    } else if (key == "session_end_s") {
      if (!parse_seconds(value))
        return fail("invalid session_end_s value '" + value + "'");
      config->session_end_s = std::strtod(value.c_str(), nullptr);
    } else if (key == "model") {
      if (value.empty()) return fail("empty model value");
      config->model = value;
    } else {
      std::fprintf(stderr,
                   "dictate: config: warning: ignoring unknown key '%s'\n",
                   key.c_str());
    }
  }
  return true;
}

// Resolve a configured model name: an absolute path is used as-is, otherwise
// the name is resolved under <executable's dir>/models/ (R8).
std::string ResolveModelPath(const std::string &name) {
  if (!name.empty() && name[0] == '/') return name;
  return ExecutableDirectory() + "/models/" + name;
}

// ---------------------------------------------------------------------------
// Idle wake-phrase detection (ticket 02) — state
// ---------------------------------------------------------------------------
// The daemon idles with the capture stream open. Idle audio is VAD-gated:
// only a *completed* speech segment (a speech run followed by 0.5 s of
// silence) is transcribed and fuzzy-matched against the wake phrase. The
// detector keeps a rolling window so a completed segment's audio can be
// extracted, and a per-run buffer (pre-roll + speech) so the segment is
// handed to whisper exactly once — never on continuous audio.
struct App;  // forward declaration (defined below)
struct WakeDetector {
  int64_t total = 0;          // idle samples received since Reset
  std::vector<int16_t> ring;  // rolling window: last kWakeRingSamples
  int64_t ring_written = 0;   // total samples written to the ring
  bool run_active = false;    // a speech frame is open in the current run
  int64_t run_speech_end = 0; // index in seg_pcm: end of the last speech frame
  int64_t run_silence = 0;    // continuous silence samples after speech
  std::vector<int16_t> seg_pcm;  // current run: pre-roll + speech (+ tail)
  int segments = 0;          // completed segments transcribed this run
  std::string last_text;     // last segment's transcript

  void Reset();
  bool Feed(App *app, const int16_t *samples, int64_t n);
  void Flush(App *app, bool *matched_out);

 private:
  void OnFrame(App *app, bool speech, int64_t frame_start, const int16_t *frame,
               int n, bool *matched_out);
  void Finalize(App *app, bool *matched_out);
};

struct App {
  // --- PipeWire objects ---
  struct pw_loop *loop = nullptr;
  struct pw_context *context = nullptr;
  struct pw_core *core = nullptr;
  struct pw_registry *registry = nullptr;
  struct pw_stream *stream = nullptr;
  // One spa_hook per listener: the PipeWire hook machinery stores the events
  // vtable by pointer, so each vtable below must have static storage.
  struct spa_hook core_hook{};
  struct spa_hook registry_hook{};
  struct spa_hook metadata_hook{};
  struct spa_hook stream_hook{};

  // --- discovery state ---
  bool metadata_bound = false;
  bool default_source_set = false;
  std::string default_source;
  std::vector<std::pair<std::string, uint32_t>> source_nodes;  // name -> node id
  bool sync_done = false;

  // --- capture state ---
  int64_t captured = 0;
  uint32_t negotiated_rate = kRequestRate;
  uint32_t negotiated_channels = kRequestChannels;
  bool stop_requested = false;
  bool stream_failed = false;
  bool session_active = false;  // persistent stream: a session is receiving samples
  std::string error_message;
  std::vector<int16_t> pcm;
  char param_buffer[512]{};  // scratch buffer for the connect-time Format pod

  // --- capture probe (temporary diagnostics) ---
  // Per-session delivery timeline: stream-state timestamps, first-sample
  // time, max tick gap (time between consecutive buffer deliveries in the
  // process callback), and a per-second sample counter. Localizes capture
  // under-delivery on the host: slow connect (late first sample), consumer
  // stall / SPA ring overrun (large tick gap), or mid-session node pause
  // (stream state leaving STREAMING).
  bool probe_active = false;
  std::chrono::steady_clock::time_point probe_t0{};
  double probe_first_sample_t = -1.0;  // s, -1 when no samples yet
  double probe_last_tick_t = -1.0;     // s
  double probe_max_tick_gap = 0.0;     // s
  int64_t probe_watermark = 0;         // captured samples at last second print
  int probe_last_sec = -1;
  // Per-second loop diagnostics (temporary): loop-iteration period, VADFeed
  // cost, and tick (OnStreamProcess call) stats for the last completed
  // second, plus session-wide tick count.
  int64_t probe_tick_count = 0;        // OnStreamProcess calls this second
  double probe_loop_period_max_ms = 0.0;  // max loop-iteration duration (ms)
  double probe_vad_total_ms = 0.0;    // VADFeed time this second (ms)
  double probe_vad_max_ms = 0.0;      // single slowest VADFeed call (ms)
  int64_t probe_iter_count = 0;       // loop iterations this second
  int64_t probe_tick_samples_max = 0; // most samples in one OnStreamProcess call

  // --- VAD / session-stop state (T-2.2) ---
  struct whisper_vad_context *vctx = nullptr;
  float vad_threshold = 0.5f;
  int64_t vad_feed_offset = 0;  // samples in pcm[] already fed to the VAD
  int64_t vad_samples_fed = 0;  // samples fed to the VAD since the last chunk
  int64_t vad_session_samples = 0;  // total VAD samples this session (timeline
                                     // for the 2 s/20 s rules; survives chunks)
  bool speech_detected = false;
  int64_t last_speech_end = 0;  // stream position just after the last speech frame
  bool vad_ok = true;
  std::string stop_reason;
  std::vector<float> vad_scratch;
  std::string input_file;
  std::string wake_test_file;  // --wake-test FILE
  std::string vad_model_path;
  std::string stt_model_path;  // --model PATH

  // --- Hotkey configuration (T-2.6, D12) ---
  // Push-to-talk chord: while all of these modifiers are held, capture;
  // releasing any of them stops the session.
  int hold_mods = kModCtrl | kModAlt;
  std::string hold_label = "ctrl+alt";

  // --- idle wake detection (ticket 02) ---
  std::vector<int16_t> idle_capture;  // idle audio since the last feed
  WakeDetector wake;

  // --- pause loop (ticket 03) ---
  bool capturing = false;        // session is receiving samples into pcm
  bool chunk_requested = false;  // pause_chunk_s pause w/ buffered speech: chunk now
  int chunks_typed = 0;          // chunks transcribed+typed this session
  int64_t session_total_samples = 0;  // all session audio, across chunks
  // Samples arriving before this steady-clock time are dropped (the session
  // start cue is still playing, so its audio must not be transcribed).
  std::chrono::steady_clock::time_point discard_until_t{};

  // --- autodictate config (ticket 04) ---
  std::string wake_phrase = kDefaultWakePhrase;
  int64_t pause_chunk_samples =
      static_cast<int64_t>(kDefaultPauseChunkSeconds *
                          static_cast<double>(kRequestRate));
  int64_t session_end_samples =
      static_cast<int64_t>(kDefaultSessionEndSeconds *
                          static_cast<double>(kRequestRate));

  // --- STT model (T-2.5, whisper.cpp) ---
  struct whisper_context *stt = nullptr;

  ~App() {
    if (stt) whisper_free(stt);
    if (vctx) whisper_vad_free(vctx);
    if (stream) pw_stream_destroy(stream);
    if (core) pw_core_destroy(core, nullptr);
    if (context) pw_context_destroy(context);
    if (loop) pw_loop_destroy(loop);
  }
};

// ---------------------------------------------------------------------------
// VAD pipeline (T-2.2)
// ---------------------------------------------------------------------------

// Feed one VAD frame (one 1280-sample / 80 ms frame, or the session's
// trailing partial) into the running Silero VAD and update the session stop
// state. The VAD runs the frame through its LSTM (state persists across
// calls) and replaces its per-call probability buffer with one speech
// probability per model window (512 samples = 32 ms for this model). The
// frame counts as speech if any window probability meets the threshold; the
// stop conditions are then evaluated at the frame's end position (audio
// timeline).
// Run one VAD frame (one 1280-sample / 80 ms frame, or the session's
// trailing partial) through the running Silero VAD and report its speech
// decision WITHOUT touching the session's stop state. The VAD runs the frame
// through its LSTM (state persists across calls) and replaces its per-call
// probability buffer with one speech probability per model window (512
// samples = 32 ms for this model). The frame counts as speech if any window
// probability meets the threshold.
bool VADDetectSpeech(App *app, const int16_t *samples, int n_samples) {
  app->vad_scratch.resize(static_cast<std::size_t>(n_samples));
  for (int i = 0; i < n_samples; ++i)
    app->vad_scratch[static_cast<std::size_t>(i)] =
        static_cast<float>(samples[i]) / 32768.0f;
  if (!whisper_vad_detect_speech_no_reset(app->vctx, app->vad_scratch.data(),
                                          n_samples)) {
    app->vad_ok = false;  // graph compute failure
    return false;
  }
  const float *probs = whisper_vad_probs(app->vctx);
  const int n_probs = whisper_vad_n_probs(app->vctx);
  bool frame_is_speech = false;
  for (int i = 0; i < n_probs; ++i)
    frame_is_speech = frame_is_speech || probs[i] >= app->vad_threshold;
  return frame_is_speech;
}

// Feed one VAD frame into the running Silero VAD and update the session stop
// state. The stop conditions are evaluated at the frame's end position
// (audio timeline).
void VADFeedFrame(App *app, const int16_t *samples, int n_samples) {
  const bool frame_is_speech = VADDetectSpeech(app, samples, n_samples);
  // The pause/session-end rules run on a session-global timeline that
  // survives chunks (ProcessChunk resets the buffer-local counters, not this
  // one), so a 20 s continuous silence is measured from the last speech
  // even if it began before the previous chunk fired.
  const int64_t frame_end = app->vad_session_samples + n_samples;
  app->vad_session_samples = frame_end;
  if (frame_is_speech) {
    app->speech_detected = true;
    app->last_speech_end = frame_end;
  }
  if (app->stop_reason.empty()) {
    const int64_t silence_run = frame_end - app->last_speech_end;
    if (app->speech_detected && silence_run >= app->pause_chunk_samples) {
      // 1. Pause chunk (ticket 03): a pause of pause_chunk_s with
      //    buffered speech — the caller transcribes+types the buffer,
      //    clears it, and resumes capturing (ProcessChunk).
      app->chunk_requested = true;
    } else if (silence_run >= app->session_end_samples) {
      // 2. Session end (ticket 03): session_end_s of continuous
      //    silence ends the session; the daemon returns to idle.
      app->stop_reason = "session end";
    }
  }
}

// Feed VAD frames from pcm[pos..pos+size). Consumes full 1280-sample frames;
// consumes the trailing partial only when allow_partial is true (session end).
// Stops consuming when the pipeline stops. Returns samples consumed.
int64_t VADFeed(App *app, const int16_t *pcm, int64_t size, bool allow_partial) {
  int64_t pos = 0;
  while (pos < size && app->vad_ok && app->stop_reason.empty()) {
    const int64_t remaining = size - pos;
    if (remaining < kVadFrameSamples && !allow_partial) break;
    const int n = static_cast<int>(std::min<int64_t>(kVadFrameSamples, remaining));
    VADFeedFrame(app, pcm + pos, n);
    pos += n;
  }
  return pos;
}

// ---------------------------------------------------------------------------
// Metadata object: "default" (published by WirePlumber)
// ---------------------------------------------------------------------------

int OnMetadataProperty(void *data, uint32_t subject, const char *key,
                       const char *type, const char *value) {
  App *app = static_cast<App *>(data);
  (void)subject;
  (void)type;
  if (key && std::strcmp(key, "default.audio.source") == 0) {
    const std::string name = ExtractSourceName(value);
    if (!name.empty()) {
      app->default_source = std::move(name);
      app->default_source_set = true;
    }
  }
  return 0;
}

const struct pw_metadata_events kMetadataEvents = {
    PW_VERSION_METADATA_EVENTS,
    OnMetadataProperty,
};

// ---------------------------------------------------------------------------
// Registry: bind the "default" metadata object; collect audio source nodes
// ---------------------------------------------------------------------------

void OnRegistryGlobal(void *data, uint32_t id, uint32_t permissions,
                      const char *type, uint32_t version,
                      const struct spa_dict *props) {
  App *app = static_cast<App *>(data);
  (void)permissions;
  if (!type) return;
  if (std::strcmp(type, "PipeWire:Interface:Node") == 0) {
    const char *name = props ? spa_dict_lookup(props, "node.name") : nullptr;
    const char *class_ = props ? spa_dict_lookup(props, "media.class") : nullptr;
    if (name && class_ && std::strcmp(class_, "Audio/Source") == 0)
      app->source_nodes.emplace_back(name, id);
  } else if (std::strcmp(type, "PipeWire:Interface:Metadata") == 0 &&
             !app->metadata_bound) {
    const char *name = props ? spa_dict_lookup(props, "metadata.name") : nullptr;
    if (name && std::strcmp(name, "default") == 0) {
      app->metadata_bound = true;
      struct pw_metadata *md =
          static_cast<struct pw_metadata *>(
              pw_registry_bind(app->registry, id, type, version, 0));
      if (md)
        pw_metadata_add_listener(md, &app->metadata_hook, &kMetadataEvents, app);
    }
  }
}

const struct pw_registry_events kRegistryEvents = {
    PW_VERSION_REGISTRY_EVENTS,
    OnRegistryGlobal,
    nullptr,  // global_remove: not needed for a short-lived capture
};

// ---------------------------------------------------------------------------
// Core: sync barrier (registry dump + bound-metadata property events flushed)
// ---------------------------------------------------------------------------

void OnCoreDone(void *data, uint32_t id, int seq) {
  App *app = static_cast<App *>(data);
  (void)seq;
  if (id == 0)
    app->sync_done = true;
}

const struct pw_core_events kCoreEvents = {
    PW_VERSION_CORE_EVENTS,
    nullptr,     // info
    OnCoreDone,  // done
    nullptr,     // ping
    nullptr,     // error
    nullptr,     // remove_id
    nullptr,     // bound_id
    nullptr,     // add_mem
    nullptr,     // remove_mem
    nullptr,     // bound_props
};

// ---------------------------------------------------------------------------
// Capture stream
// ---------------------------------------------------------------------------

// Wall seconds since the capture probe started for the current session, or
// -1.0 when the probe is inactive.
double ProbeTime(App *app) {
  if (!app->probe_active) return -1.0;
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - app->probe_t0)
      .count();
}

void PrintProbeSummary(App *app) {
  // Probe summary: when the first sample arrived (connect speed) and the
  // worst tick gap (consumer stall / ring overrun) during the session.
  if (app->probe_first_sample_t < 0.0)
    std::printf("dictate: probe: no samples delivered, max tick gap %.3f s\n",
                app->probe_max_tick_gap);
  else
    std::printf("dictate: probe: first sample t+%0.3f s, max tick gap %.3f s\n",
                app->probe_first_sample_t, app->probe_max_tick_gap);
}

void OnStreamStateChanged(void *data, enum pw_stream_state old_state,
                          enum pw_stream_state state, const char *error) {
  App *app = static_cast<App *>(data);
  (void)old_state;
  (void)error;
  if (state == PW_STREAM_STATE_ERROR) {
    app->stream_failed = true;
    app->error_message = error ? error : "unknown stream error";
  }
  // Probe: log stream state transitions (skips the trivial UNCONNECTED
  // initial state) so a failing capture is visible in the probe summary.
  if (app->probe_active && state != PW_STREAM_STATE_UNCONNECTED)
    std::fprintf(stderr, "  [probe] stream state=%d\n", static_cast<int>(state));
}

void OnStreamParamChanged(void *data, uint32_t id, const struct spa_pod *param) {
  App *app = static_cast<App *>(data);
  if (!param) return;  // end-of-params sentinel
  if (id == SPA_PARAM_Format) {
    struct spa_audio_info_raw info = {};
    if (spa_format_audio_raw_parse(param, &info) == 0) {
      if (info.rate != 0) app->negotiated_rate = info.rate;
      if (info.channels != 0) app->negotiated_channels = info.channels;
    }
  }
}

void OnStreamProcess(void *data) {
  App *app = static_cast<App *>(data);
  // The capture stream is persistent: every buffer is drained each tick. A
  // session receives samples into pcm only while it is capturing; while a
  // pause chunk is transcribing, incoming audio is dropped (ticket 03), and
  // idle samples feed the wake detector.
  const bool active = app->session_active && app->capturing;
  if (app->probe_active) app->probe_tick_count++;
  int64_t call_samples = 0;
  struct pw_buffer *buffer;
  while ((buffer = pw_stream_dequeue_buffer(app->stream)) != nullptr) {
    if (active) {
      const struct spa_buffer *sb = buffer->buffer;
      if (sb && sb->n_datas >= 1 && sb->datas[0].data && sb->datas[0].chunk) {
        const struct spa_data *d = &sb->datas[0];
        uint32_t size = d->chunk->size & ~1u;  // S16: keep byte count even
        if (size > 0) {
          // Drop audio while the session-start cue is still playing: it would
          // otherwise be captured and transcribed as dictation (the cue says
          // "Hi Kunal, I'm listening."). See StartSession for the window.
          const bool discard =
              std::chrono::steady_clock::now() < app->discard_until_t;
          if (!discard) {
            uint32_t maxsize = d->maxsize ? d->maxsize : size;
            uint32_t offset = d->chunk->offset % maxsize;
            const int64_t nsamples = static_cast<int64_t>(size / 2);
            // The session has no hard cap: grow the capture buffer on demand.
            if (app->pcm.size() <
                static_cast<std::size_t>(app->captured + nsamples))
              app->pcm.resize(static_cast<std::size_t>(app->captured + nsamples));
            int16_t *dst = app->pcm.data() + app->captured;
            const uint8_t *src = static_cast<const uint8_t *>(d->data);
            if (offset + size <= maxsize) {
              std::memcpy(dst, src + offset, size);
            } else {  // ring-buffer wrap
              const std::size_t first = maxsize - offset;
              std::memcpy(dst, src + offset, first);
              std::memcpy(dst + first / 2, src, size - first);
            }
            app->captured += nsamples;
            app->session_total_samples += nsamples;
            call_samples += nsamples;
            if (app->probe_active && app->probe_first_sample_t < 0.0)
              app->probe_first_sample_t = ProbeTime(app);
          }
        }
      }
    } else if (app->session_active) {
      // Session active but capture paused (a chunk is transcribing):
      // incoming audio is dropped (ticket 03, accepted constraint).
    } else {
      // Idle: keep the latest audio for wake-phrase detection (the daemon
      // loop feeds it through the VAD gate — see WakeDetector). The buffer
      // is cleared after every feed, so it only ever holds what arrived
      // since the last loop iteration; a stuck loop is detected by the cap.
      const struct spa_buffer *sb = buffer->buffer;
      if (sb && sb->n_datas >= 1 && sb->datas[0].data && sb->datas[0].chunk) {
        const struct spa_data *d = &sb->datas[0];
        uint32_t size = d->chunk->size & ~1u;  // S16: keep byte count even
        if (size > 0) {
          uint32_t maxsize = d->maxsize ? d->maxsize : size;
          uint32_t offset = d->chunk->offset % maxsize;
          const int64_t nsamples = static_cast<int64_t>(size / 2);
          if (static_cast<int64_t>(app->idle_capture.size()) + nsamples >
              kWakeRingSamples)
            app->idle_capture.clear();  // stale: the loop is stuck
          const uint8_t *src = static_cast<const uint8_t *>(d->data);
          if (offset + size <= maxsize) {
            const int16_t *p = reinterpret_cast<const int16_t *>(src + offset);
            app->idle_capture.insert(app->idle_capture.end(), p,
                                     p + nsamples);
          } else {  // ring-buffer wrap
            const int64_t first = static_cast<int64_t>((maxsize - offset) / 2);
            const int16_t *p = reinterpret_cast<const int16_t *>(src + offset);
            const int16_t *q = reinterpret_cast<const int16_t *>(src);
            app->idle_capture.insert(app->idle_capture.end(), p, p + first);
            app->idle_capture.insert(app->idle_capture.end(), q,
                                     q + nsamples - first);
          }
        }
      }
    }
    pw_stream_queue_buffer(app->stream, buffer);
  }
  if (app->probe_active && call_samples > app->probe_tick_samples_max)
    app->probe_tick_samples_max = call_samples;
  // Probe: one tick per buffer batch delivered to this callback. A large gap
  // means the consumer stopped draining the SPA ring (which then overruns and
  // silently drops samples).
  if (app->probe_active) {
    const double t = ProbeTime(app);
    if (t >= 0.0) {
      if (app->probe_last_tick_t >= 0.0) {
        const double gap = t - app->probe_last_tick_t;
        if (gap > app->probe_max_tick_gap) app->probe_max_tick_gap = gap;
      }
      app->probe_last_tick_t = t;
    }
  }
}

const struct pw_stream_events kStreamEvents = {
    PW_VERSION_STREAM_EVENTS,
    nullptr,              // destroy
    OnStreamStateChanged,  // state_changed
    nullptr,              // control_info
    nullptr,              // io_changed
    OnStreamParamChanged,  // param_changed
    nullptr,              // add_buffer
    nullptr,              // remove_buffer
    OnStreamProcess,      // process
    nullptr, nullptr, nullptr,  // drained, command, trigger_done
};

// ---------------------------------------------------------------------------
// Minimal 16-bit PCM WAV writer (RIFF/"fmt "/data, little-endian)
// ---------------------------------------------------------------------------

bool WriteWav(const char *path, const std::vector<int16_t> &pcm,
              uint32_t rate, uint32_t channels) {
  std::FILE *f = std::fopen(path, "wb");
  if (!f) return false;
  auto wle = [](std::FILE *file, auto value) {
    using T = std::decay_t<decltype(value)>;
    uint8_t buf[sizeof(T)];
    for (std::size_t i = 0; i < sizeof(T); ++i)
      buf[i] = static_cast<uint8_t>(value >> (i * 8));
    return std::fwrite(buf, 1, sizeof(T), file) == sizeof(T);
  };
  const uint32_t data_size = static_cast<uint32_t>(pcm.size()) * 2u;
  bool ok = std::fwrite("RIFF", 4, 1, f) == 1 && wle(f, 36u + data_size) &&
            std::fwrite("WAVE", 4, 1, f) == 1 && std::fwrite("fmt ", 4, 1, f) == 1 &&
            wle(f, 16u) && wle(f, uint16_t{1}) &&
            wle(f, static_cast<uint16_t>(channels)) && wle(f, rate) &&
            wle(f, rate * channels * 2u) &&
            wle(f, static_cast<uint16_t>(channels * 2)) && wle(f, uint16_t{16}) &&
            std::fwrite("data", 4, 1, f) == 1 && wle(f, data_size) &&
            std::fwrite(pcm.data(), 2, pcm.size(), f) == pcm.size();
  std::fclose(f);
  return ok;
}

// ---------------------------------------------------------------------------
// Minimal 16-bit PCM WAV reader (RIFF/"fmt "/data, little-endian)
// ---------------------------------------------------------------------------

bool ReadWav16(const char *path, std::vector<int16_t> *samples,
               uint32_t *rate, uint32_t *channels, std::string *error) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f) {
    *error = std::string("cannot open '") + path + "'";
    return false;
  }
  auto fail = [&](const char *msg) {
    std::fclose(f);
    *error = msg;
    return false;
  };

  long file_size = 0;
  if (std::fseek(f, 0, SEEK_END) != 0 || (file_size = std::ftell(f)) < 12 ||
      std::fseek(f, 0, SEEK_SET) != 0) {
    return fail("not a valid WAV file (truncated?)");
  }

  uint8_t riff[12] = {};
  if (std::fread(riff, 1, 12, f) != 12 || std::memcmp(riff, "RIFF", 4) != 0 ||
      std::memcmp(riff + 8, "WAVE", 4) != 0) {
    return fail("not a RIFF/WAVE file");
  }

  const auto le16 = [](const uint8_t *p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
  };
  const auto le32 = [](const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
  };

  bool have_fmt = false, have_data = false, ok = true;
  uint32_t fmt_format = 0, fmt_channels = 0, fmt_rate = 0, fmt_bits = 0;
  long pos = 12;
  while (pos + 8 <= file_size && ok) {
    uint8_t chunk[8] = {};
    if (std::fseek(f, pos, SEEK_SET) != 0 || std::fread(chunk, 1, 8, f) != 8) {
      ok = false;
      break;
    }
    const uint32_t chunk_size = le32(chunk + 4);
    const long payload_pos = pos + 8;
    const long read_bytes =
        std::min<long>(static_cast<long>(chunk_size), file_size - payload_pos);
    if (std::memcmp(chunk, "fmt ", 4) == 0) {
      uint8_t fmt[16] = {};
      if (read_bytes >= 16 && std::fseek(f, payload_pos, SEEK_SET) == 0 &&
          std::fread(fmt, 1, 16, f) == 16) {
        fmt_format = le16(fmt);
        fmt_channels = le16(fmt + 2);
        fmt_rate = le32(fmt + 4);
        fmt_bits = le16(fmt + 14);
        have_fmt = true;
      } else {
        ok = false;
      }
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      const uint32_t n_samples = static_cast<uint32_t>(read_bytes / 2);
      samples->resize(n_samples);
      if (n_samples > 0 && std::fseek(f, payload_pos, SEEK_SET) == 0 &&
          std::fread(samples->data(), 2, n_samples, f) != n_samples) {
        ok = false;
      }
      have_data = true;
    }
    pos = payload_pos + static_cast<long>(chunk_size) + (chunk_size & 1u);
  }
  if (!ok) return fail("I/O error while reading WAV");
  if (!have_fmt) return fail("WAV is missing the 'fmt ' chunk");
  if (!have_data) return fail("WAV is missing the 'data' chunk");
  if (fmt_format != 1) return fail("unsupported WAV audio format (need PCM format 1)");
  if (fmt_bits != 16) return fail("unsupported WAV bit depth (need 16)");

  *rate = fmt_rate;
  *channels = fmt_channels;
  std::fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// Key watch (T-2.3, T-2.6): evdev observation of the configured hotkeys
// ---------------------------------------------------------------------------

// A push-to-talk event detected by KeyWatcher: start when the whole
// configured chord is pressed, stop when any chord key is released; t_rel is
// seconds since the watch began (the timestamp of the triggering key event).
struct KeyEvent {
  bool start;
  double t_rel;
  std::string device;
};

// Whether a device can produce a chord press for the given modifier-group
// mask: it must expose at least one key from each configured group (left or
// right variant).
bool HasModifierKeys(struct libevdev *dev, int mods) {
  if ((mods & kModCtrl) != 0 &&
      libevdev_has_event_code(dev, EV_KEY, KEY_LEFTCTRL) == 0 &&
      libevdev_has_event_code(dev, EV_KEY, KEY_RIGHTCTRL) == 0)
    return false;
  if ((mods & kModShift) != 0 &&
      libevdev_has_event_code(dev, EV_KEY, KEY_LEFTSHIFT) == 0 &&
      libevdev_has_event_code(dev, EV_KEY, KEY_RIGHTSHIFT) == 0)
    return false;
  if ((mods & kModAlt) != 0 &&
      libevdev_has_event_code(dev, EV_KEY, KEY_LEFTALT) == 0 &&
      libevdev_has_event_code(dev, EV_KEY, KEY_RIGHTALT) == 0)
    return false;
  if ((mods & kModSuper) != 0 &&
      libevdev_has_event_code(dev, EV_KEY, KEY_LEFTMETA) == 0 &&
      libevdev_has_event_code(dev, EV_KEY, KEY_RIGHTMETA) == 0)
    return false;
  return true;
}

// Observation-only push-to-talk keyboard watcher: devices are opened
// read-only and never grabbed (no EVIOCGRAB), so keys keep reaching the
// focused application.
//
// scan() enumerates /dev/input/event* and keeps devices that report EV_KEY
// and expose the configured modifier keys (left or right of each configured
// group); a watch/skip line is printed for every candidate. Per-device
// modifier state is a held counter per group (value 1 press / value 0
// release, clamped at zero; auto-repeat value 2 is ignored). A press (value
// 1) that completes the configured chord (the device's held modifier groups
// are exactly the configured ones) is a start event; a release (value 0) of
// any chord key (the held groups were exactly the configured ones before the
// release) is a stop event: while all chord keys are held the session is
// capturing, and letting go of any of them ends it. An extra held modifier
// (e.g. Super, a KDE shortcut) never completes the chord.
//
// The object owns its fds (libevdev_free() does not close them; they are
// closed here) and exposes them via fdOf() so the daemon can poll() them
// alongside PipeWire in a single loop (T-2.5).
class KeyWatcher {
 public:
  KeyWatcher(bool verbose, int hold_mods)
      : verbose_(verbose), hold_mods_(hold_mods) {}

  // Enumerate and open /dev/input/event* devices; returns the number kept.
  // A missing /dev/input (e.g. container environments) is not an error.
  int scan();

  int deviceCount() const { return static_cast<int>(devices_.size()); }

  // The pollable fd of device i (close only through KeyWatcher destruction).
  int fdOf(int i) const { return devices_[i].fd; }

  // Drain device i's buffered events (the fd is O_NONBLOCK, so
   // libevdev_next_event returns -EAGAIN once the buffer is empty),
  // update the Ctrl state, print per the verbosity, and append detected
  // start/stop events to out_events. Safe to call even when the fd is not
  // ready.
  void consume(int i, std::vector<KeyEvent> *out_events);

  // Total raw keydowns (EV_KEY, value 1) seen since scan().
  int keydownCount() const { return keydowns_; }

  // Seconds since scan() began.
  double time() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         t0_)
        .count();
  }

 private:
  struct Device {
    int fd = -1;  // owned; closed on destruction
    struct libevdev *dev = nullptr;  // freed on destruction
    std::string name;
    int ctrl_held = 0;   // Ctrl keys currently down on this device
    int meta_held = 0;   // Super/Meta keys currently down on this device
    int alt_held = 0;    // Alt keys currently down on this device
    int shift_held = 0;  // Shift keys currently down on this device

    Device() = default;
    // Transfer fd/libevdev ownership; the source becomes inert.
    Device(Device &&o) noexcept
        : fd(o.fd), dev(o.dev), name(std::move(o.name)),
          ctrl_held(o.ctrl_held), meta_held(o.meta_held),
          alt_held(o.alt_held), shift_held(o.shift_held) {
      o.fd = -1;
      o.dev = nullptr;
    }
    Device &operator=(Device &&o) noexcept {
      if (this != &o) {
        if (dev) libevdev_free(dev);
        if (fd >= 0) ::close(fd);
        fd = o.fd;
        dev = o.dev;
        name = std::move(o.name);
        ctrl_held = o.ctrl_held;
        meta_held = o.meta_held;
        alt_held = o.alt_held;
        shift_held = o.shift_held;
        o.fd = -1;
        o.dev = nullptr;
      }
      return *this;
    }
    ~Device() {
      if (dev) libevdev_free(dev);  // does not close the fd
      if (fd >= 0) ::close(fd);
    }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
  };

  bool verbose_;
  int hold_mods_;
  std::vector<Device> devices_;
  int keydowns_ = 0;
  std::chrono::steady_clock::time_point t0_;
};

int KeyWatcher::scan() {
  t0_ = std::chrono::steady_clock::now();
  devices_.clear();

  std::vector<std::string> candidates;
  DIR *dir = ::opendir("/dev/input");
  if (!dir) {
    std::fprintf(stderr, "dictate: cannot open /dev/input: %s\n",
                 std::strerror(errno));
    return 0;
  }
  struct dirent *entry;
  while ((entry = ::readdir(dir)) != nullptr) {
    const std::string name = entry->d_name;
    if (name.rfind("event", 0) != 0) continue;  // /dev/input/eventN only
    candidates.push_back("/dev/input/" + name);
  }
  ::closedir(dir);
  std::sort(candidates.begin(), candidates.end());

  for (const std::string &path : candidates) {
    // O_NONBLOCK: libevdev_next_event follows the fd's blocking state; a
    // blocking fd would make consume() block on the read() instead of
    // draining the buffer and returning (which would hang the poll loop and
    // defeat the watch window).
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
      std::fprintf(stderr, "dictate: skip %s: open failed: %s\n", path.c_str(),
                   std::strerror(errno));
      continue;
    }
    struct libevdev *raw = nullptr;
    const int rc = libevdev_new_from_fd(fd, &raw);
    if (rc != 0) {
      // libevdev does not take the fd on failure; close it ourselves.
      std::fprintf(stderr, "dictate: skip %s: %s\n", path.c_str(),
                   std::strerror(-rc));
      ::close(fd);
      continue;
    }
    devices_.emplace_back();
    Device &d = devices_.back();
    d.fd = fd;
    d.dev = raw;
    d.name = libevdev_get_name(raw);
    const bool has_key = libevdev_has_event_type(raw, EV_KEY) != 0;
    const bool has_mods = HasModifierKeys(raw, hold_mods_);
    if (!has_key || !has_mods) {
      std::fprintf(stderr, "dictate: skip %s (%s): %s\n", path.c_str(),
                   d.name.c_str(),
                   !has_key ? "no EV_KEY events"
                            : "no configured modifier keys");
      devices_.pop_back();
      continue;
    }
    std::printf("dictate: watch %s (%s)\n", path.c_str(), d.name.c_str());
  }
  return static_cast<int>(devices_.size());
}

void KeyWatcher::consume(int i, std::vector<KeyEvent> *out_events) {
  Device &d = devices_[i];
  struct input_event ev{};
  while (true) {
    const int rc = libevdev_next_event(d.dev, LIBEVDEV_READ_FLAG_NORMAL, &ev);
    if (rc == LIBEVDEV_READ_STATUS_SYNC) continue;  // dropped diff, continue
    if (rc != LIBEVDEV_READ_STATUS_SUCCESS) return;  // -EAGAIN (drained) or error
    if (ev.type != EV_KEY) continue;
    const bool is_ctrl =
        (ev.code == KEY_LEFTCTRL || ev.code == KEY_RIGHTCTRL);
    const bool is_meta = (ev.code == KEY_LEFTMETA || ev.code == KEY_RIGHTMETA);
    const bool is_alt = (ev.code == KEY_LEFTALT || ev.code == KEY_RIGHTALT);
    const bool is_shift =
        (ev.code == KEY_LEFTSHIFT || ev.code == KEY_RIGHTSHIFT);
    if (ev.value == 1) {  // press (auto-repeat repeats are value 2)
      if (is_ctrl) d.ctrl_held++;
      if (is_meta) d.meta_held++;
      if (is_alt) d.alt_held++;
      if (is_shift) d.shift_held++;
      keydowns_++;
      if (verbose_) {
        const char *name = libevdev_event_code_get_name(EV_KEY, ev.code);
        std::printf("  [t+%7.3f s] keydown %-12s (%s)\n", time(),
                    name ? name : "unknown", d.name.c_str());
      }
      // Push-to-talk: a press can only complete the chord, so if the held
      // modifier groups on this device are exactly the configured ones, the
      // session starts (an extra held modifier, e.g. Super, does not fire it).
      const int held = (d.ctrl_held > 0 ? kModCtrl : 0) |
                       (d.meta_held > 0 ? kModSuper : 0) |
                       (d.alt_held > 0 ? kModAlt : 0) |
                       (d.shift_held > 0 ? kModShift : 0);
      if (held == hold_mods_) {
        std::printf("  [t+%7.3f s] start (%s)\n", time(), d.name.c_str());
        if (out_events) {
          KeyEvent ke;
          ke.start = true;
          ke.t_rel = time();
          ke.device = d.name;
          out_events->push_back(ke);
        }
      }
    } else if (ev.value == 0) {  // release
      // Push-to-talk: releasing any chord key ends the session — the held
      // groups are checked before the decrement below.
      const int held = (d.ctrl_held > 0 ? kModCtrl : 0) |
                       (d.meta_held > 0 ? kModSuper : 0) |
                       (d.alt_held > 0 ? kModAlt : 0) |
                       (d.shift_held > 0 ? kModShift : 0);
      if (held == hold_mods_) {
        std::printf("  [t+%7.3f s] stop (%s)\n", time(), d.name.c_str());
        if (out_events) {
          KeyEvent ke;
          ke.start = false;
          ke.t_rel = time();
          ke.device = d.name;
          out_events->push_back(ke);
        }
      }
      if (is_ctrl && d.ctrl_held > 0) d.ctrl_held--;
      if (is_meta && d.meta_held > 0) d.meta_held--;
      if (is_alt && d.alt_held > 0) d.alt_held--;
      if (is_shift && d.shift_held > 0) d.shift_held--;
    }
    // value == 2 (auto-repeat): no state change, no event
  }
}

// ---------------------------------------------------------------------------
// Session tail: magnitude stats, WAV write, capture summary, VAD report
// ---------------------------------------------------------------------------

int FinishSession(App *app, bool live, const std::string &source,
                  uint32_t node_id) {
  std::vector<int16_t> &pcm = app->pcm;
  if (pcm.empty()) {
    std::fprintf(stderr, "dictate: session produced no audio\n");
    return 1;
  }

  // Sample statistics over |sample| (magnitude).
  uint32_t min_mag = UINT32_MAX, max_mag = 0;
  double sum_mags = 0.0;
  for (const int16_t s : pcm) {
    const uint32_t m =
        s < 0 ? static_cast<uint32_t>(-static_cast<int>(s)) : static_cast<uint32_t>(s);
    min_mag = std::min(min_mag, m);
    max_mag = std::max(max_mag, m);
    sum_mags += static_cast<double>(m);
  }
  const double mean_mag = sum_mags / static_cast<double>(pcm.size());

  if (!WriteWav(kWavPath, pcm, app->negotiated_rate, app->negotiated_channels)) {
    std::fprintf(stderr, "dictate: failed to write %s\n", kWavPath);
    return 1;
  }

  const double captured_seconds =
      static_cast<double>(pcm.size()) / static_cast<double>(app->negotiated_rate);
  if (live) {
    std::printf(
        "dictate: captured %.3f s (%lld samples), %u Hz, %u ch, node: %s (id %u)\n",
        captured_seconds, static_cast<long long>(pcm.size()),
        app->negotiated_rate, app->negotiated_channels, source.c_str(), node_id);
  } else {
    std::printf(
        "dictate: captured %.3f s (%lld samples), %u Hz, %u ch, input: %s\n",
        captured_seconds, static_cast<long long>(pcm.size()),
        app->negotiated_rate, app->negotiated_channels, source.c_str());
  }
  std::printf("dictate: sample magnitude: min=%u max=%u mean=%.1f\n", min_mag, max_mag,
              mean_mag);
  if (max_mag == 0)
    std::fprintf(stderr,
                 "dictate: warning: captured audio is pure silence — "
                 "check the system mic mute/volume\n");

  // Speech segments over the session's audio: whisper.cpp re-runs the VAD
  // over the full session with a fresh LSTM state (clean window alignment)
  // and applies its default segmenter (min speech 250 ms, min silence 100
  // ms, 30 ms padding). The segment getters return centiseconds, so divide
  // by 100 for seconds.
  const whisper_vad_params vad_params = whisper_vad_default_params();
  app->vad_scratch.resize(pcm.size());
  for (std::size_t i = 0; i < pcm.size(); ++i)
    app->vad_scratch[i] = static_cast<float>(pcm[i]) / 32768.0f;
  struct whisper_vad_segments *segments = whisper_vad_segments_from_samples(
      app->vctx, vad_params, app->vad_scratch.data(),
      static_cast<int>(pcm.size()));
  const int n_segments = segments ? whisper_vad_segments_n_segments(segments) : 0;
  std::printf("dictate: VAD: threshold %.2f, %d speech segment(s)\n",
              app->vad_threshold, n_segments);
  for (int i = 0; i < n_segments; ++i) {
    const float t0 = whisper_vad_segments_get_segment_t0(segments, i) / 100.0f;
    const float t1 = whisper_vad_segments_get_segment_t1(segments, i) / 100.0f;
    std::printf("dictate:   segment %d: start %.3f s, end %.3f s, duration %.3f s\n",
                i, t0, t1, t1 - t0);
  }
  if (segments) whisper_vad_free_segments(segments);

  std::printf("dictate: total audio processed: %.3f s (%lld samples)\n",
              static_cast<double>(app->vad_samples_fed) /
                  static_cast<double>(kRequestRate),
              static_cast<long long>(app->vad_samples_fed));
  std::printf("dictate: stop reason: %s\n", app->stop_reason.c_str());
  return 0;
}

// ---------------------------------------------------------------------------
// Per-session capture (T-2.5)
// ---------------------------------------------------------------------------

// Reset the per-session capture/VAD state for a new dictation session. The
// pcm buffer keeps its allocated capacity across sessions (no per-session
// memory growth); the VAD LSTM state is reset between utterances so each
// session's speech detection starts clean.
void ResetSession(App *app) {
  app->pcm.clear();
  app->captured = 0;
  app->stop_requested = false;
  app->stream_failed = false;
  app->error_message.clear();
  app->vad_feed_offset = 0;
  app->vad_samples_fed = 0;
  app->speech_detected = false;
  app->last_speech_end = 0;
  app->stop_reason.clear();
  app->capturing = false;
  app->chunk_requested = false;
  app->chunks_typed = 0;
  app->session_total_samples = 0;
  app->vad_session_samples = 0;
  if (app->vctx) whisper_vad_reset_state(app->vctx);
}

// Establish the persistent PipeWire connection: event loop, context, core,
// and the core/registry listeners, then wait for the "default" metadata
// global (bounded) and the registry sync barrier. Called once at startup; the
// connection, and the capture stream built on it (SetupCaptureStream), stay
// alive for the daemon's lifetime.
bool SetupPipeWire(App *app) {
  struct pw_loop *loop = pw_loop_new(nullptr);
  if (!loop) {
    std::fprintf(stderr, "dictate: failed to create PipeWire event loop\n");
    return false;
  }
  struct pw_context *context = pw_context_new(loop, nullptr, 0);
  struct pw_core *core = context ? pw_context_connect(context, nullptr, 0) : nullptr;
  if (!core) {
    std::fprintf(stderr,
                 "dictate: failed to connect to PipeWire "
                 "(is the session manager running?)\n");
    pw_context_destroy(context);
    pw_loop_destroy(loop);
    return false;
  }
  app->loop = loop;
  app->context = context;
  app->core = core;
  app->registry = pw_core_get_registry(core, PW_VERSION_REGISTRY, 0);

  pw_core_add_listener(core, &app->core_hook, &kCoreEvents, app);
  pw_registry_add_listener(app->registry, &app->registry_hook, &kRegistryEvents, app);

  const auto now = [] { return std::chrono::steady_clock::now(); };

  // Phase 1: wait for the WirePlumber "default" metadata global (it is
  // enumerated during the connection handshake; bounded wait covers setups
  // without WirePlumber).
  auto deadline = now() + std::chrono::duration<double>(kMetadataTimeout);
  while (!app->metadata_bound && now() < deadline)
    pw_loop_iterate(loop, 100);

  // Phase 2: sync barrier — guarantees the registry dump is complete and the
  // property events for the bound metadata object have been delivered.
  app->sync_done = false;
  pw_core_sync(core, 0, 0);
  deadline = now() + std::chrono::duration<double>(kSyncTimeout);
  while (!app->sync_done && now() < deadline)
    pw_loop_iterate(loop, 100);
  if (!app->sync_done) {
    std::fprintf(stderr, "dictate: timed out waiting for PipeWire registry sync\n");
    return false;
  }
  return true;
}

// Resolve the target source node from the registry state: the WirePlumber
// "default" audio source if published, else the first physical ALSA input
// node (fallback warning kept).
bool ResolveSource(App *app, std::string *target, uint32_t *node_id) {
  if (app->default_source_set) {
    *target = app->default_source;
    for (const auto &s : app->source_nodes)
      if (s.first == *target) {
        *node_id = s.second;
        break;
      }
  } else {
    // Fallback: prefer a physical ALSA input source, else any Audio/Source node.
    std::size_t pick = std::string::npos;
    for (std::size_t i = 0; i < app->source_nodes.size(); ++i)
      if (app->source_nodes[i].first.rfind("alsa_input.", 0) == 0) {
        pick = i;
        break;
      }
    if (pick == std::string::npos && !app->source_nodes.empty()) pick = 0;
    if (pick == std::string::npos) {
      std::fprintf(stderr,
                   "dictate: no default input found (no WirePlumber 'default' "
                   "source and no Audio/Source node in the registry)\n");
      return false;
    }
    *target = app->source_nodes[pick].first;
    *node_id = app->source_nodes[pick].second;
    std::fprintf(stderr,
                 "dictate: warning: no WirePlumber 'default' source; using "
                 "fallback source node '%s'\n",
                 target->c_str());
  }
  return true;
}

// Create and connect the persistent capture stream (16 kHz mono S16_LE). The
// stream stays connected for the lifetime of the daemon: creating it per
// session cost ~0.7 s of connect/negotiate time, dropping the first ~0.7 s
// of each utterance (the "missing first word" from T-2.5). Buffers arriving
// outside an active session are drained and discarded (OnStreamProcess).
bool SetupCaptureStream(App *app, const std::string &target) {
  // The negotiated format starts at the requested values; the Format param
  // event at connect time updates it.
  app->negotiated_rate = kRequestRate;
  app->negotiated_channels = kRequestChannels;

  // Stream properties: a capture stream must declare itself as
  // "Stream/Input/Audio". WirePlanner derives the session item direction from
  // media.class; "Audio/Source" would mark this item as a device source and
  // the select-target chain would look for targets in the wrong direction.
  struct pw_properties *props = pw_properties_new(
      PW_KEY_MEDIA_NAME, "dictate",
      PW_KEY_MEDIA_CLASS, "Stream/Input/Audio",
      PW_KEY_NODE_NAME, "dictate",
      PW_KEY_TARGET_OBJECT, target.c_str(),
      nullptr);
  app->stream = pw_stream_new(app->core, "dictate-capture", props);
  if (!app->stream) {
    std::fprintf(stderr, "dictate: failed to create capture stream\n");
    return false;
  }
  pw_stream_add_listener(app->stream, &app->stream_hook, &kStreamEvents, app);

  // Declare the desired format at connect time (16 kHz mono S16_LE). The pod
  // must carry the SPA_PARAM_EnumFormat id: WirePlanner's audio adapter
  // enumerates the node's EnumFormat capabilities when it configures the
  // session item, and a plain SPA_PARAM_Format pod is invisible to that query
  // (the adapter then reports "no usable format found" and links nothing).
  struct spa_pod_builder builder =
      SPA_POD_BUILDER_INIT(app->param_buffer, sizeof(app->param_buffer));
  struct spa_audio_info_raw info = {};
  info.format = SPA_AUDIO_FORMAT_S16_LE;
  info.rate = kRequestRate;
  info.channels = kRequestChannels;
  info.position[0] = SPA_AUDIO_CHANNEL_MONO;
  struct spa_pod *format_pod =
      spa_format_audio_raw_build(&builder, SPA_PARAM_EnumFormat, &info);
  const struct spa_pod *connect_params[] = {format_pod};
  // No PW_STREAM_FLAG_MAP_BUFFERS: buffers are auto-mapped on dequeue, so
  // datas[0].data is directly usable in the process callback.
  if (pw_stream_connect(app->stream, PW_DIRECTION_INPUT, PW_ID_ANY,
                        PW_STREAM_FLAG_AUTOCONNECT, connect_params, 1) < 0) {
    std::fprintf(stderr, "dictate: failed to start capture stream\n");
    pw_stream_disconnect(app->stream);
    pw_stream_destroy(app->stream);
    app->stream = nullptr;
    return false;
  }
  return true;
}

// Read a PCM WAV file's duration in seconds by parsing its RIFF chunks.
// Returns 0.0 on any read/parse failure (caller falls back to a default).
double ReadWavDurationSeconds(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return 0.0;
  char riff[12];
  in.read(riff, 12);
  if (in.gcount() != 12 || std::memcmp(riff, "RIFF", 4) != 0 ||
      std::memcmp(riff + 8, "WAVE", 4) != 0)
    return 0.0;
  uint32_t sample_rate = 0, channels = 0, bits = 0, data_size = 0;
  bool have_fmt = false, have_data = false;
  char hdr[8];
  while (in) {
    in.read(hdr, 8);
    if (in.gcount() != 8) break;
    uint32_t csz;
    in.read(reinterpret_cast<char *>(&csz), 4);
    if (in.gcount() != 4) break;
    if (std::memcmp(hdr, "fmt ", 4) == 0) {
      uint16_t audio_fmt;
      in.read(reinterpret_cast<char *>(&audio_fmt), 2);
      in.read(reinterpret_cast<char *>(&channels), 2);
      in.read(reinterpret_cast<char *>(&sample_rate), 4);
      char skip[6];  // byte rate (4) + block align (2)
      in.read(skip, 6);
      in.read(reinterpret_cast<char *>(&bits), 2);
      have_fmt = true;
    } else if (std::memcmp(hdr, "data", 4) == 0) {
      data_size = csz;
      have_data = true;
      break;
    } else {
      in.seekg(static_cast<std::streamoff>(csz), std::ios::cur);
    }
  }
  if (!have_fmt || !have_data || sample_rate == 0 || channels == 0 || bits == 0)
    return 0.0;
  const double bytes_per_sec =
      static_cast<double>(sample_rate) * static_cast<double>(channels) *
      static_cast<double>(bits / 8);
  return static_cast<double>(data_size) / bytes_per_sec;
}

// Play the session-start cue (ticket 04): <executable's dir>/cue.wav via
// paplay, spawned in a detached child process so it never blocks the
// capture loop. A missing cue.wav degrades to a one-line stderr warning;
// a missing/unavailable paplay dies silently in the child — never fatal.
void PlayCue() {
  const std::string path = ExecutableDirectory() + "/cue.wav";
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) {
    std::fprintf(stderr, "dictate: cue: %s not found — no cue played\n",
                 path.c_str());
    return;
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    std::fprintf(stderr, "dictate: cue: fork failed: %s\n",
                 std::strerror(errno));
    return;
  }
  if (pid == 0) {
    // Child: detach stderr so a paplay failure never pollutes the daemon
    // log, then exec. paplay missing → silent _exit(127).
    const int devnull = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (devnull >= 0) {
      ::dup2(devnull, STDERR_FILENO);
      ::close(devnull);
    }
    const char *args[3] = {"paplay", path.c_str(), nullptr};
    ::execvp("paplay", const_cast<char *const *>(args));
    _exit(127);
  }
}

// Start a new dictation session on the persistent capture stream: reset the
// per-session state and begin receiving samples.
void StartSession(App *app) {
  ResetSession(app);
  app->session_active = true;
  app->capturing = true;
  // Session-start cue (ticket 04): play cue.wav detached — it never blocks
  // this path.
  PlayCue();

  // The mic picks up the cue ("Hi Kunal, I'm listening.") while it plays; we
  // must not transcribe it. Drop session audio until the cue has finished:
  // its real duration (from the WAV header) plus a margin for paplay startup
  // latency and any speech tail. A missing/unreadable cue.wav falls back to a
  // conservative default so the cue window is still covered.
  const std::string cue_path = ExecutableDirectory() + "/cue.wav";
  double cue_s = ReadWavDurationSeconds(cue_path);
  if (cue_s <= 0.0) cue_s = 2.5;  // fallback if the header can't be read
  app->discard_until_t = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(cue_s + kCueDiscardMarginSeconds));

  // Capture probe: timeline for this session (stream state, first sample,
  // tick gaps, per-second sample counts).
  app->probe_active = true;
  // Unbuffer stdout so every diagnostic line reaches the log immediately
  // (a killed daemon must not lose buffered probe lines).
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  app->probe_t0 = std::chrono::steady_clock::now();
  app->probe_first_sample_t = -1.0;
  app->probe_last_tick_t = -1.0;
  app->probe_max_tick_gap = 0.0;
  app->probe_watermark = 0;
  app->probe_last_sec = -1;
  app->probe_iter_count = 0;
  app->probe_tick_count = 0;
  app->probe_loop_period_max_ms = 0.0;
  app->probe_vad_total_ms = 0.0;
  app->probe_vad_max_ms = 0.0;

  // Wake detection resumes clean after every session (the session's VAD
  // LSTM state was already reset by ResetSession above).
  app->wake.Reset();
  app->idle_capture.clear();

}

// End the active session: assign the stop reason, feed the trailing partial
// VAD frame (at most 1279 samples) that the loop could not consume, and run
// the session tail (stats, WAV, VAD report). Returns 0 when the session audio
// is ready for transcription, 1 on a session failure (an error is printed).
int EndSession(App *app, bool stop_key, const std::string &target,
               uint32_t node_id) {
  app->session_active = false;
  if (stop_key && app->stop_reason.empty()) app->stop_reason = "stop key";
  // Skipped when the pipeline already stopped (e.g. 20 s silence) so the
  // session never feeds audio past its stop point.
  if (app->vad_ok && app->stop_reason.empty()) {
    app->vad_feed_offset +=
        VADFeed(app, app->pcm.data() + app->vad_feed_offset,
                app->captured - app->vad_feed_offset, true);
  }
  if (app->stop_reason.empty()) app->stop_reason = "session end";
  if (!app->vad_ok) {
    std::fprintf(stderr, "dictate: VAD computation failed\n");
    return 1;
  }

  // Probe summary: when the first sample arrived and the worst tick gap
  // (consumer stall / ring overrun) during the session.
  PrintProbeSummary(app);

  app->pcm.resize(static_cast<std::size_t>(
      std::min(app->captured, app->vad_samples_fed)));
  if (app->pcm.empty()) {
    // Clean session end (ticket 03): every pause chunk was already
    // transcribed+typed as it fired, so there is nothing left to report as
    // audio.
    std::printf("dictate: session: stop reason: %s, chunks typed: %d, "
                "total audio %.3f s\n",
                app->stop_reason.c_str(), app->chunks_typed,
                static_cast<double>(app->session_total_samples) /
                    static_cast<double>(kRequestRate));
    return 0;
  }
  return FinishSession(app, true, target, node_id);
}

// ---------------------------------------------------------------------------
// Typing via uinput (T-2.4)
// ---------------------------------------------------------------------------

// Character-to-scan-code mapping, US layout. Each entry is the *base* scan
// code plus whether Shift is held (e.g. '!' is KEY_1 + Shift) — that is what
// the compositor's keymap understands. Scan codes are NOT consecutive across
// the alphabet (home row a–l is 30–38; bottom row z–m is 44–50; digits are
// 2–11), so every character is mapped explicitly. This table is the single
// source of truth: TypeChunk looks characters up here and declares every code
// the table uses — the kernel drops events for codes a device doesn't
// declare.
struct CharKey {
  char c;
  int code;
  bool shift;
};

constexpr CharKey kCharKeys[] = {
    {' ', KEY_SPACE, false},
    {'\n', KEY_ENTER, false},
    // Top row (q–p).
    {'q', KEY_Q, false},
    {'w', KEY_W, false},
    {'e', KEY_E, false},
    {'r', KEY_R, false},
    {'t', KEY_T, false},
    {'y', KEY_Y, false},
    {'u', KEY_U, false},
    {'i', KEY_I, false},
    {'o', KEY_O, false},
    {'p', KEY_P, false},
    // Home row (a–l): consecutive.
    {'a', KEY_A, false},
    {'s', KEY_S, false},
    {'d', KEY_D, false},
    {'f', KEY_F, false},
    {'g', KEY_G, false},
    {'h', KEY_H, false},
    {'j', KEY_J, false},
    {'k', KEY_K, false},
    {'l', KEY_L, false},
    // Bottom row (z–m): a separate block, not contiguous with the home row.
    {'z', KEY_Z, false},
    {'x', KEY_X, false},
    {'c', KEY_C, false},
    {'v', KEY_V, false},
    {'b', KEY_B, false},
    {'n', KEY_N, false},
    {'m', KEY_M, false},
    // Digits.
    {'1', KEY_1, false},
    {'2', KEY_2, false},
    {'3', KEY_3, false},
    {'4', KEY_4, false},
    {'5', KEY_5, false},
    {'6', KEY_6, false},
    {'7', KEY_7, false},
    {'8', KEY_8, false},
    {'9', KEY_9, false},
    {'0', KEY_0, false},
    // Punctuation and symbols.
    {'.', KEY_DOT, false},
    {',', KEY_COMMA, false},
    {'/', KEY_SLASH, false},
    {';', KEY_SEMICOLON, false},
    {':', KEY_SEMICOLON, true},
    {'\'', KEY_APOSTROPHE, false},
    {'"', KEY_APOSTROPHE, true},
    {'!', KEY_1, true},
    {'?', KEY_SLASH, true},
    {'(', KEY_9, true},
    {')', KEY_0, true},
    {'-', KEY_MINUS, false},
    {'_', KEY_MINUS, true},
    {'+', KEY_EQUAL, true},
    {'=', KEY_EQUAL, false},
    {'\\', KEY_BACKSLASH, false},
    {'|', KEY_BACKSLASH, true},
    {'`', KEY_GRAVE, false},
    {'~', KEY_GRAVE, true},
    {'[', KEY_LEFTBRACE, false},
    {']', KEY_RIGHTBRACE, false},
    {'{', KEY_LEFTBRACE, true},
    {'}', KEY_RIGHTBRACE, true},
    {'@', KEY_2, true},
    {'#', KEY_3, true},
    {'$', KEY_4, true},
    {'%', KEY_5, true},
    {'^', KEY_6, true},
    {'&', KEY_7, true},
    {'*', KEY_8, true},
    {'<', KEY_COMMA, true},
    {'>', KEY_DOT, true},
};

// Look up c in kCharKeys. Uppercase letters use the lowercase key plus Shift.
// Sets code and shift; code is -1 when c is unmapped.
void CharToKeyCode(char c, int &code, bool &shift) {
  const char base = (c >= 'A' && c <= 'Z')
                        ? static_cast<char>(c - 'A' + 'a')
                        : c;
  code = -1;
  shift = (c >= 'A' && c <= 'Z');
  for (const CharKey &ck : kCharKeys)
    if (ck.c == base) {
      code = ck.code;
      shift = shift || ck.shift;
      return;
    }
}

// Hold each key for this long between keydown and keyup. Machine-speed
// bursts (a whole sentence emitted back-to-back in one input batch) get
// coalesced/rate-dropped by the compositor input pipeline — live host test
// 2026-09-02: ~50% of keystrokes dropped on 40–110 char outputs. A short
// human-plausible dwell makes every keypress its own input batch.
const int kTypeDwellUs = 5 * 1000;

// A uinput keyboard device (T-2.4). Open() creates and registers the
// device; TypeChunk() types text into it; the destructor destroys the device
// (UI_DEV_DESTROY + closing the fd). A streaming typing run keeps one
// device open across all of its chunks, so the compositor sees a single
// fast typist rather than many short-lived ones.
struct TypedKeyboard {
  int fd = -1;
  bool ok = true;

  void Open() {
    fd = ::open("/dev/uinput", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      ok = false;
      std::fprintf(stderr,
                   "dictate: cannot open /dev/uinput (%s); the uinput kernel "
                   "module is likely not loaded (try: modprobe uinput) — text "
                   "not typed\n",
                   std::strerror(errno));
      return;
    }
    // Register a keyboard-only uinput device: identity + capabilities, then
    // UI_DEV_CREATE (destroyed by UI_DEV_DESTROY / closing the fd).
    struct uinput_setup setup{};
    setup.id.bustype = BUS_USB;
    setup.id.vendor = 0x0001;
    setup.id.product = 0x0002;
    setup.id.version = 0x0001;
    std::snprintf(setup.name, sizeof(setup.name), "%s", "dictate-uinput");
    if (::ioctl(fd, UI_DEV_SETUP, &setup) != 0 ||
        ::ioctl(fd, UI_SET_EVBIT, EV_KEY) != 0)
      ok = false;
    // Declare every scan code the table uses (plus Shift): the kernel drops
    // events for codes the device doesn't declare, so an incomplete
    // declaration silently loses characters.
    for (const CharKey &ck : kCharKeys)
      if (::ioctl(fd, UI_SET_KEYBIT, ck.code) != 0)
        ok = false;
    if (::ioctl(fd, UI_SET_KEYBIT, KEY_LEFTSHIFT) != 0)
      ok = false;
    if (::ioctl(fd, UI_SET_KEYBIT, KEY_BACKSPACE) != 0)
      ok = false;
    if (ok && ::ioctl(fd, UI_DEV_CREATE) != 0)
      ok = false;

    // Give the compositor a moment to discover the new device before the
    // first keypress, otherwise early keys may be dropped (evdev has no
    // queue for readers that are not open yet).
    usleep(100 * 1000);
  }

  ~TypedKeyboard() {
    if (fd >= 0) {
      ::ioctl(fd, UI_DEV_DESTROY);
      ::close(fd);
    }
  }

  // Type one chunk of text. Returns the number of characters actually
  // emitted (unmapped characters are skipped), or -1 when the device is not
  // open; the per-key dwell (D11) makes each keypress its own compositor
  // input batch.
  int TypeChunk(const std::string &text) {
    if (fd < 0) return -1;
    int typed = 0;
    auto emit = [&](int type, int code, int value) {
      struct input_event ev{};
      ev.type = type;
      ev.code = code;
      ev.value = value;
      if (::write(fd, &ev, sizeof(ev)) != static_cast<ssize_t>(sizeof(ev)))
        ok = false;
    };
    auto report = [&]() { emit(EV_SYN, SYN_REPORT, 0); };
    for (char c : text) {
      int code = -1;
      bool shift = false;
      CharToKeyCode(c, code, shift);
      if (code < 0) continue;  // Unmapped character: skip.
      ++typed;
      if (shift) {
        emit(EV_KEY, KEY_LEFTSHIFT, 1);
        report();
        usleep(kTypeDwellUs);
      }
      emit(EV_KEY, code, 1);
      report();
      usleep(kTypeDwellUs);
      emit(EV_KEY, code, 0);
      report();
      if (shift) {
        emit(EV_KEY, KEY_LEFTSHIFT, 0);
        report();
      }
    }
    return ok ? typed : -1;
  }

  // Delete the last `count` characters (backspaces), used by the T-2.7 tail
  // fix to replace the last typed chunks with a corrected retyping.
  bool Backspace(int count) {
    for (int i = 0; i < count; ++i) {
      struct input_event down{};
      down.type = EV_KEY;
      down.code = KEY_BACKSPACE;
      down.value = 1;
      struct input_event up{};
      up.type = EV_KEY;
      up.code = KEY_BACKSPACE;
      up.value = 0;
      struct input_event syn{};
      syn.type = EV_SYN;
      syn.code = SYN_REPORT;
      syn.value = 0;
      if (::write(fd, &down, sizeof(down)) != static_cast<ssize_t>(sizeof(down)) ||
          ::write(fd, &up, sizeof(up)) != static_cast<ssize_t>(sizeof(up)) ||
          ::write(fd, &syn, sizeof(syn)) != static_cast<ssize_t>(sizeof(syn)))
        return false;
      usleep(kTypeDwellUs);
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// Transcription + typing (T-2.5)
// ---------------------------------------------------------------------------

// Clean a whisper transcription for typing: trim leading/trailing whitespace
// (including newlines). Punctuation is kept as-is; the whisper segment
// getters return plain text without timestamp tokens.
std::string CleanTranscription(const std::string &text) {
  const std::size_t begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const std::size_t end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1);
}

// Sentinel segments whisper emits for unusable audio (blank / non-English /
// phonetics-only); a session whose segments are only these is "no speech".
const char *const kSentinels[] = {"[BLANK_AUDIO]", "[NON-ENGLISH SPEECH]",
                                  "[PHONETICS]"};

bool IsSentinelSegment(const std::string &text) {
  const std::string cleaned = CleanTranscription(text);
  for (const char *sentinel : kSentinels)
    if (cleaned == sentinel) return true;
  return false;
}

// T-2.7: stream-typing chunk size (whisper_full_params.max_tokens). whisper
// forces a chunk boundary every N tokens and fires a new_segment callback per
// chunk, so completed chunks are typed while the decode still runs. N=8 is
// the smallest chunk size whose boundaries stay clean: N=4-7 lose words
// (whisper.cpp's single-timestamp seek skip swallows the following audio) and
// N=16+ duplicates boundary words.
const int kStreamChunkTokens = 8;

// One streaming typing run (T-2.7): owns the uinput keyboard (opened lazily
// on the first typed chunk) and accumulates the typed text (st.full) plus
// per-chunk lengths/start offsets for the tail fix.
struct StreamTyper {
  TypedKeyboard keyboard;
  bool keyboard_opened = false;
  bool first_chunk = true;
  int chunks_typed = 0;
  std::vector<std::string> typed;  // chunk texts as typed (verbatim)
  std::vector<int> chunk_lens;     // characters actually emitted per chunk
  std::vector<int64_t> chunk_t0;   // each chunk's audio start (10ms units)
  std::string full;                // concatenation of the typed chunks
};

// whisper.cpp new_segment callback: types each completed chunk as it appears
// (sentinel chunks skipped), lazily opening the keyboard on the first typed
// chunk. Leading whitespace is trimmed on the first typed chunk only; later
// chunk texts are typed verbatim.
void OnNewSegment(whisper_context *ctx, whisper_state *state, int n_new,
                  void *user_data) {
  (void)ctx;
  StreamTyper *st = static_cast<StreamTyper *>(user_data);
  if (!st) return;
  const int n_total = whisper_full_n_segments_from_state(state);
  for (int i = n_total - n_new; i < n_total; ++i) {
    std::string chunk = whisper_full_get_segment_text_from_state(state, i);
    if (IsSentinelSegment(chunk)) continue;
    if (st->first_chunk) {
      const std::size_t b = chunk.find_first_not_of(" \t\r\n");
      chunk = (b == std::string::npos) ? std::string{} : chunk.substr(b);
      if (chunk.empty()) continue;
      st->first_chunk = false;
    }
    if (!st->keyboard_opened) {
      st->keyboard.Open();
      st->keyboard_opened = true;
    }
    const int n = st->keyboard.TypeChunk(chunk);
    st->typed.push_back(chunk);
    st->chunk_lens.push_back(n < 0 ? 0 : n);
    st->chunk_t0.push_back(whisper_full_get_segment_t0_from_state(state, i));
    st->full += chunk;
    st->chunks_typed++;
    std::printf("dictate: typed: %s\n", chunk.c_str());
  }
}

// ---------------------------------------------------------------------------
// Wake-phrase detection (ticket 02) — matcher, segment transcription,
// and the WakeDetector methods declared above the App struct.
// ---------------------------------------------------------------------------

// Lowercase a string, keeping only letters (punctuation/spaces stripped).
std::string NormalizeForMatch(const std::string &text) {
  std::string out;
  for (char c : text)
    if (std::isalpha(static_cast<unsigned char>(c)))
      out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

// Levenshtein edit distance (two-row DP).
int Levenshtein(const std::string &a, const std::string &b) {
  if (a.empty()) return static_cast<int>(b.size());
  if (b.empty()) return static_cast<int>(a.size());
  std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
  for (std::size_t j = 0; j <= b.size(); ++j)
    prev[j] = static_cast<int>(j);
  for (std::size_t i = 1; i <= a.size(); ++i) {
    cur[0] = static_cast<int>(i);
    for (std::size_t j = 1; j <= b.size(); ++j)
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
    std::swap(prev, cur);
  }
  return prev[b.size()];
}

// Decide whether a transcript is the wake phrase.
//
// Tolerance design — the evidence: Kunal's real "hi piya" recording
// transcribes as "Hypia. Apia. Apia. Hypia.", so a whole-string Levenshtein
// check would fail it (the repeat makes the string far longer than the
// phrase). The rule is therefore:
//   1. any single word within Levenshtein <= kWakeMaxEdits of the
//      normalized phrase ("hipiya") matches — a mangled single utterance
//      ("hypia", d(hipiya)=2) triggers, while an ordinary sentence, whose
//      words are all far from the phrase, never does;
//   2. the whole normalized transcript within the same distance also
//      matches (clean "hi piya" / "hipiya");
//   3. a length cap: the normalized transcript may not exceed 3x the phrase
//      length, so long utterances are rejected outright.
bool IsWakePhrase(const std::string &wake_phrase, const std::string &raw) {
  const std::string target = NormalizeForMatch(wake_phrase);
  std::string joined;
  std::string word;
  auto word_match = [&] {
    if (!word.empty() && Levenshtein(word, target) <= kWakeMaxEdits)
      return true;
    word.clear();
    return false;
  };
  for (char c : raw) {
    if (std::isalpha(static_cast<unsigned char>(c))) {
      const char l =
          static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      joined += l;
      word += l;
    } else if (word_match()) {
      return true;
    }
  }
  if (word_match()) return true;
  if (joined.size() > 3 * target.size()) return false;
  return Levenshtein(joined, target) <= kWakeMaxEdits;
}

// Transcribe an arbitrary PCM buffer (no typing, no tail fix): peak-
// normalized greedy whisper decode; sentinel segments dropped. Returns the
// joined, trimmed segment text; false only on a decode failure.
bool TranscribePcm(App *app, const std::vector<int16_t> &pcm,
                   std::string *out) {
  *out = "";
  if (!app->stt) {
    std::fprintf(stderr, "dictate: STT model not loaded\n");
    return false;
  }
  if (pcm.empty()) return true;
  int peak = 0;
  for (const int16_t s : pcm)
    peak = std::max(peak, s < 0 ? -s : s);
  const float scale = (peak > 0) ? (0.95f / (peak / 32768.0f)) : 1.0f;
  std::vector<float> f32(pcm.size());
  for (std::size_t i = 0; i < f32.size(); ++i)
    f32[i] = static_cast<float>(pcm[i]) * scale / 32768.0f;
  struct whisper_full_params params =
      whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  params.print_progress = false;
  if (whisper_full(app->stt, params, f32.data(),
                   static_cast<int>(f32.size())) != 0) {
    std::fprintf(stderr, "dictate: wake: transcription failed\n");
    return false;
  }
  std::string text;
  for (int i = 0; i < whisper_full_n_segments(app->stt); ++i) {
    const std::string seg = whisper_full_get_segment_text(app->stt, i);
    if (IsSentinelSegment(seg)) continue;
    const std::string t = CleanTranscription(seg);
    if (t.empty()) continue;
    if (!text.empty()) text += " ";
    text += t;
  }
  *out = text;
  return true;
}

void WakeDetector::Reset() {
  total = 0;
  ring.clear();
  ring_written = 0;
  run_active = false;
  run_speech_end = 0;
  run_silence = 0;
  seg_pcm.clear();
  segments = 0;
  last_text.clear();
}

// Handle one VAD frame's decision. frame_start is the global sample index of
// the frame's first sample (global coordinates span Feed calls, so the ring
// — not the caller's chunk — is the source for pre-roll).
void WakeDetector::OnFrame(App *app, bool speech, int64_t frame_start,
                           const int16_t *frame, int n, bool *matched_out) {
  const int64_t max_samples =
      static_cast<int64_t>(kWakeMaxSeconds * kRequestRate);
  if (speech) {
    run_silence = 0;
    if (!run_active) {
      run_active = true;
      seg_pcm.clear();
      // Pre-roll: up to one frame of audio before the first speech frame,
      // from the rolling window (when still available).
      const int64_t ring_floor = ring_written - kWakeRingSamples;
      const int64_t pre_start =
          std::max(ring_floor, frame_start - kVadFrameSamples);
      for (int64_t g = pre_start; g < frame_start; ++g)
        seg_pcm.push_back(ring[g % kWakeRingSamples]);
      run_speech_end = static_cast<int64_t>(seg_pcm.size());
    } else if (static_cast<int64_t>(seg_pcm.size()) > max_samples) {
      // A long speech run: ordinary speech, not a phrase. Drop it — it is
      // never transcribed, which also bounds the buffer.
      run_active = false;
      seg_pcm.clear();
      *matched_out = false;
      return;
    }
    seg_pcm.insert(seg_pcm.end(), frame, frame + n);
    run_speech_end = static_cast<int64_t>(seg_pcm.size());
  } else if (run_active) {
    run_silence += n;
    if (run_silence <= kWakeMaxSilenceSamples)
      seg_pcm.insert(seg_pcm.end(), frame, frame + n);
    if (run_silence >= kWakeMaxSilenceSamples) {
      run_active = false;
      run_silence = 0;
      Finalize(app, matched_out);
    }
  }
}

// A speech run has ended (0.5 s of trailing silence, or input ended via
// Flush). Extract the speech portion (pre-roll + speech, silence tail
// trimmed) and, only when its length is in the 0.4–6 s window where whisper
// is trustworthy and a phrase is plausible, transcribe and match it.
void WakeDetector::Finalize(App *app, bool *matched_out) {
  *matched_out = false;
  last_text.clear();
  std::vector<int16_t> seg;
  seg.resize(static_cast<std::size_t>(run_speech_end));
  for (std::size_t i = 0; i < seg.size(); ++i)
    seg[i] = seg_pcm[i];
  seg_pcm.clear();
  const int64_t min_samples =
      static_cast<int64_t>(kWakeMinSeconds * kRequestRate);
  const int64_t max_samples =
      static_cast<int64_t>(kWakeMaxSeconds * kRequestRate);
  const int64_t len = static_cast<int64_t>(seg.size());
  if (len < min_samples || len > max_samples) return;  // not a phrase candidate
  std::string text;
  if (!TranscribePcm(app, seg, &text)) return;
  segments++;
  last_text = text;
  *matched_out = IsWakePhrase(app->wake_phrase, text);
}

// Append a chunk of idle audio to the rolling window and run every full VAD
// frame in it through the shared Silero VAD. Returns true as soon as a
// completed speech segment matches the wake phrase.
bool WakeDetector::Feed(App *app, const int16_t *samples, int64_t n) {
  if (n <= 0) return false;
  ring.resize(kWakeRingSamples);
  for (int64_t i = 0; i < n; ++i)
    ring[(ring_written + i) % kWakeRingSamples] = samples[i];
  ring_written += n;
  total += n;
  int64_t pos = 0;
  while (pos + kVadFrameSamples <= n && app->vad_ok) {
    const bool speech = VADDetectSpeech(app, samples + pos, kVadFrameSamples);
    bool matched = false;
    OnFrame(app, speech, total - n + pos, samples + pos, kVadFrameSamples,
            &matched);
    if (matched) return true;
    pos += kVadFrameSamples;
  }
  return false;
}

// End of input (wake-test only): a run open at the file's end is a completed
// segment — the file cannot provide the trailing silence.
void WakeDetector::Flush(App *app, bool *matched_out) {
  if (!run_active) return;
  run_active = false;
  run_silence = 0;
  Finalize(app, matched_out);
}

// Run whisper.cpp (greedy sampling, CPU-only) over the session's PCM,
// streaming completed chunks into `st` through the new_segment callback (see
// above), then repair the tail: the last chunk's forced boundary can clip
// trailing tokens (probe: "factors." -> "fact."), so the last two chunks'
// region is re-decoded unforced (plain decode = baseline quality) and, if it
// differs from the typed tail, the typed tail is deleted and the corrected
// text retyped. Returns false (and prints an error) when the STT model is
// missing or a run fails.
bool Transcribe(App *app, const std::vector<int16_t> &pcm, StreamTyper *st) {
  if (!app->stt) {
    std::fprintf(stderr, "dictate: STT model not loaded\n");
    return false;
  }
  // Peak-normalize the capture for whisper: system mic levels vary wildly
  // (a quiet USB mic can be ~10x below typical whisper input; a mic at full
  // gain clips), and whisper mishears quiet audio badly. Scale the loudest
  // sample to ~0.95 so the model sees near-full-scale speech.
  int peak = 0;
  for (const int16_t s : pcm) {
    const int m = s < 0 ? -s : s;
    peak = std::max(peak, m);
  }
  const float scale = (peak > 0) ? (0.95f / (peak / 32768.0f)) : 1.0f;
  std::vector<float> f32(pcm.size());
  for (std::size_t i = 0; i < f32.size(); ++i)
    f32[i] = static_cast<float>(pcm[i]) * scale / 32768.0f;
  struct whisper_full_params params =
      whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  params.print_progress = false;
  params.max_tokens = kStreamChunkTokens;
  params.split_on_word = true;
  params.new_segment_callback = OnNewSegment;
  params.new_segment_callback_user_data = st;
  if (whisper_full(app->stt, params, f32.data(),
                   static_cast<int>(f32.size())) != 0) {
    std::fprintf(stderr, "dictate: transcription failed\n");
    return false;
  }
  // Tail fix (T-2.7): re-decode the last two typed chunks' region unforced
  // (whisper_full reinitializes its state per call, so the tail decode
  // replaces the chunked decode's segments). Skip the fix when it produced
  // no usable text or text identical to the typed tail.
  if (st->typed.size() >= 2) {
    const int64_t t0 = st->chunk_t0[st->chunk_t0.size() - 2];
    const std::size_t s0 =
        static_cast<std::size_t>(t0) * (kRequestRate / 100);  // 10ms units
    if (s0 < f32.size()) {
      struct whisper_full_params tail_params =
          whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
      tail_params.print_progress = false;
      if (whisper_full(app->stt, tail_params, f32.data() + s0,
                       static_cast<int>(f32.size() - s0)) == 0) {
        std::string tail;
        for (int i = 0; i < whisper_full_n_segments(app->stt); ++i) {
          const std::string seg = whisper_full_get_segment_text(app->stt, i);
          if (IsSentinelSegment(seg)) continue;
          tail += seg;
        }
        while (!tail.empty() &&
               (tail.back() == ' ' || tail.back() == '\t' ||
                tail.back() == '\r' || tail.back() == '\n'))
          tail.pop_back();
        const int n = static_cast<int>(st->typed.size());
        const std::string last_two = st->typed[n - 2] + st->typed[n - 1];
        if (!tail.empty() && tail != last_two) {
          st->keyboard.Backspace(st->chunk_lens[n - 2] + st->chunk_lens[n - 1]);
          const int retyped = st->keyboard.TypeChunk(tail);
          st->full =
              st->full.substr(0, st->full.size() - last_two.size()) + tail;
          std::printf("dictate: tail: corrected: %s\n", tail.c_str());
          if (retyped < 0)
            std::fprintf(stderr, "dictate: tail fix: retyping failed\n");
        }
      }
    }
  }
  return true;
}

// Transcribe a finished session's audio and type it into the focused
// application (T-2.5, streamed per-chunk by T-2.7). The typer owns the
// keyboard and the typed text for the whole run. A transcription or typing
// failure is non-fatal: a diagnostic is printed and the session summary is
// still printed.
void TranscribeAndType(App *app) {
  const double audio_seconds =
      static_cast<double>(app->pcm.size()) /
      static_cast<double>(app->negotiated_rate);
  // Minimum-audio guard: skip transcription/typing when the session captured
  // too little audio for whisper to be trusted (it hallucinates on short
  // input). The session summary is still printed.
  if (audio_seconds < kMinAudioSeconds) {
    std::printf("dictate: skipping transcription: captured audio too short "
                "(%.3f s < %.3f s)\n",
                audio_seconds, kMinAudioSeconds);
    std::printf("dictate: session: stop reason: %s, audio %.3f s, "
                "transcribe 0.000 s\n",
                app->stop_reason.c_str(), audio_seconds);
    return;
  }
  StreamTyper typer;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = Transcribe(app, app->pcm, &typer);
  const double transcribe_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  if (!ok) {
    std::printf("dictate: session: stop reason: %s, audio %.3f s, "
                "transcribe %.3f s\n",
                app->stop_reason.c_str(), audio_seconds, transcribe_seconds);
    return;
  }
  const std::string text = CleanTranscription(typer.full);
  if (text.empty()) {
    std::printf("dictate: no speech, nothing typed\n");
  }
  std::printf("dictate: session: stop reason: %s, audio %.3f s, "
              "transcribe %.3f s\n",
              app->stop_reason.c_str(), audio_seconds, transcribe_seconds);
}

// One pause chunk (ticket 03): take the buffered audio out of the capture
// buffer, transcribe it, and type it. The buffer is cleared and the capture
// counters reset so the session resumes capturing immediately. The VAD
// timeline (last_speech_end) is NOT reset: the 20 s session-end rule counts
// continuous silence across chunks. A transcription failure is non-fatal:
// it is printed and the session continues.
void ProcessChunk(App *app) {
  app->chunk_requested = false;
  std::vector<int16_t> chunk = std::move(app->pcm);
  app->pcm.clear();
  app->captured = 0;
  app->vad_feed_offset = 0;
  app->vad_samples_fed = 0;
  app->speech_detected = false;
  // last_speech_end / vad_session_samples are intentionally NOT reset: the
  // 20 s session-end rule counts continuous silence across chunks.
  if (chunk.empty()) return;
  // Pure-silence chunk (max magnitude 0): whisper would only emit sentinels;
  // warn and type nothing.
  bool any = false;
  for (const int16_t s : chunk) {
    if (s != 0) {
      any = true;
      break;
    }
  }
  if (!any) {
    std::fprintf(stderr,
                 "dictate: warning: chunk is pure silence — nothing typed\n");
    return;
  }
  const double audio_seconds =
      static_cast<double>(chunk.size()) / static_cast<double>(kRequestRate);
  // Minimum-audio guard: whisper hallucinates on very short input.
  if (audio_seconds < kMinAudioSeconds) {
    std::printf("dictate: chunk: skipping transcription: audio too short "
                "(%.3f s < %.3f s)\n", audio_seconds, kMinAudioSeconds);
    return;
  }
  StreamTyper typer;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = Transcribe(app, chunk, &typer);
  const double transcribe_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  const std::string text = CleanTranscription(typer.full);
  if (ok && text.empty())
    std::printf("dictate: chunk: no speech, nothing typed\n");
  if (ok && !text.empty()) app->chunks_typed++;
  std::printf("dictate: chunk: audio %.3f s, transcribe %.3f s\n",
              audio_seconds, transcribe_seconds);
}

// Dictation daemon: the default mode. Idles listening for the wake phrase
// "hi piya" (ticket 02); a match opens a capture session that runs the
// pause loop (ticket 03): 2 s pauses produce typed chunks, 20 s of
// continuous silence ends the session. Per-session failures (stream, VAD,
// transcription, typing) print an error and return to idle; the daemon only
// exits on a startup failure (PipeWire, source resolution).
int RunDictationDaemon(App *app) {
  std::printf("dictate: dictation mode: say \"%s\" to start "
              "(push-to-talk chord: hold %s)\n",
              app->wake_phrase.c_str(), app->hold_label.c_str());

  // The typing path needs the uinput kernel module; warn once at startup when
  // it is unavailable (non-fatal: capture and transcription still work).
  if (::access("/dev/uinput", F_OK) != 0)
    std::fprintf(stderr,
                 "dictate: warning: uinput unavailable (try: modprobe "
                 "uinput)\n");

  KeyWatcher watcher(false, app->hold_mods);
  const int n_devices = watcher.scan();
  std::printf("dictate: watching %d keyboard device(s)\n", n_devices);

  if (!SetupPipeWire(app)) return 1;

  std::string target;
  uint32_t node_id = 0;
  if (!ResolveSource(app, &target, &node_id)) return 1;

  // The persistent capture stream: created once at startup, kept streaming
  // for the daemon's lifetime (see SetupCaptureStream).
  if (!SetupCaptureStream(app, target)) return 1;

  // The PipeWire loop fd joins the poll set: the loop must be woken by the
  // stream's own data, not paced by the keyboard poll's timeout (the T-2.5
  // root cause — ~10 loop iterations/s captured only ~21% of nominal).
  std::vector<struct pollfd> pfds;
  for (int i = 0; i < n_devices; ++i) {
    struct pollfd pfd{};
    pfd.fd = watcher.fdOf(i);
    pfd.events = POLLIN;
    pfds.push_back(pfd);
  }
  struct pollfd loop_pfd{};
  loop_pfd.fd = pw_loop_get_fd(app->loop);
  loop_pfd.events = POLLIN;
  pfds.push_back(loop_pfd);

  const auto now = [] { return std::chrono::steady_clock::now(); };
  while (true) {
    const auto iter_start = now();
    const int rc = ::poll(pfds.data(), pfds.size(), 100);
    if (rc < 0) {
      std::fprintf(stderr, "dictate: poll error: %s\n", std::strerror(errno));
      return 1;
    }
    bool start_key = false;
    bool stop_key = false;
    for (int i = 0; i < n_devices; ++i) {
      if (pfds[i].revents & POLLIN) {
        std::vector<KeyEvent> events;
        watcher.consume(i, &events);
        for (const KeyEvent &ke : events) {
          if (ke.start) start_key = true;
          if (!ke.start) stop_key = true;
        }
      }
    }
    // Always drain the PipeWire loop, regardless of the poll result.
    if (pw_loop_iterate(app->loop, 0) < 0) {
      std::fprintf(stderr, "dictate: event loop error\n");
      app->stream_failed = true;
    }

    // A dead stream is discarded and recreated (any in-flight session is
    // abandoned with its probe summary printed); the daemon retries until it
    // recovers.
    if (app->stream_failed) {
      if (app->session_active) {
        std::fprintf(stderr, "dictate: stream error: %s\n",
                     app->error_message.c_str());
        PrintProbeSummary(app);
        app->session_active = false;
        app->probe_active = false;
      }
      pw_stream_disconnect(app->stream);
      pw_stream_destroy(app->stream);
      app->stream = nullptr;
      app->stream_failed = false;
      app->error_message.clear();
      // The stream's audio history is gone: the wake detector's rolling
      // window and run state are stale.
      app->wake.Reset();
      app->idle_capture.clear();
      while (!SetupCaptureStream(app, target))
        ::poll(nullptr, 0, 100);  // backoff; the daemon survives a dead stream
    }

    if (app->session_active) {
      if (app->vad_ok) {
        const auto vad_start = now();
        app->vad_feed_offset +=
            VADFeed(app, app->pcm.data() + app->vad_feed_offset,
                    app->captured - app->vad_feed_offset, false);
        if (app->probe_active) {
          const double vad_ms =
              std::chrono::duration<double, std::milli>(now() - vad_start).count();
          app->probe_vad_total_ms += vad_ms;
          if (vad_ms > app->probe_vad_max_ms) app->probe_vad_max_ms = vad_ms;
        }
        if (!app->stop_reason.empty()) app->stop_requested = true;
        // Pause chunk (ticket 03): capture off, transcribe+type the
        // buffered audio, clear the buffer, resume capturing immediately.
        if (app->chunk_requested) {
          app->capturing = false;
          ProcessChunk(app);
          app->capturing = true;
        }
      }
      // Probe: report each completed second of wall time (how many samples
      // actually arrived during it). A healthy session prints ~100% each line.
      if (app->probe_active) {
        const double period_ms =
            std::chrono::duration<double, std::milli>(now() - iter_start).count();
        if (period_ms > app->probe_loop_period_max_ms)
          app->probe_loop_period_max_ms = period_ms;
        app->probe_iter_count++;
        const int sec = static_cast<int>(ProbeTime(app));
        if (sec > app->probe_last_sec) {
          const int64_t delta = app->captured - app->probe_watermark;
          std::printf("  [probe t+%ds] %lld samples (%.1f%% of nominal)\n", sec,
                      static_cast<long long>(delta),
                      100.0 * static_cast<double>(delta) /
                          static_cast<double>(app->negotiated_rate));
          std::printf(
              "  [probe loop t+%ds] iters %lld, period max %.1f ms, vad total "
              "%.1f ms (max %.1f ms), ticks %lld\n",
              sec, static_cast<long long>(app->probe_iter_count),
              app->probe_loop_period_max_ms, app->probe_vad_total_ms,
              app->probe_vad_max_ms,
              static_cast<long long>(app->probe_tick_count));
          app->probe_watermark = app->captured;
          app->probe_last_sec = sec;
          app->probe_iter_count = 0;
          app->probe_loop_period_max_ms = 0.0;
          app->probe_vad_total_ms = 0.0;
          app->probe_vad_max_ms = 0.0;
          app->probe_tick_count = 0;
        }
      }
      // Stop conditions: 20 s continuous silence (VAD) or the legacy
      // push-to-talk chord release.
      if (stop_key || app->stop_requested) {
        app->probe_active = false;
        const int erc = EndSession(app, stop_key, target, node_id);
        if (erc == 0 && !app->pcm.empty()) TranscribeAndType(app);
      }
    } else {
      // Idle: wake-phrase detection on the incoming audio. The VAD gates
      // everything — whisper only runs on a completed speech segment
      // (WakeDetector::Feed), never on continuous idle audio. A phrase
      // spoken during an active session never reaches this branch, so it
      // is ignored by construction.
      //
      // The live stream delivers buffers smaller than one VAD frame (the
      // source emits ~21 ms / 341-sample buffers, kVadFrameSamples is 1280).
      // WakeDetector::Feed only runs a VAD frame when it receives at least
      // 1280 samples, so feeding a 341-sample chunk runs zero frames and
      // starves the detector. Accumulate until we have a full frame, then
      // feed only whole frames (keeping any remainder for next iteration).
      if (app->idle_capture.size() >= static_cast<std::size_t>(kVadFrameSamples)) {
        const std::size_t to_feed =
            (app->idle_capture.size() / static_cast<std::size_t>(kVadFrameSamples)) *
            static_cast<std::size_t>(kVadFrameSamples);
        const bool wake = app->wake.Feed(app, app->idle_capture.data(),
                                         static_cast<int64_t>(to_feed));
        app->idle_capture.erase(app->idle_capture.begin(),
                                app->idle_capture.begin() + static_cast<std::ptrdiff_t>(to_feed));
        if (wake && app->stream) {
          std::printf("\xF0\x9F\x8E\x99 capture started\n");
          StartSession(app);
        }
      }
      // A start key begins a session on the persistent stream.
      if (start_key && app->stream && !app->session_active) {
        StartSession(app);
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  App app;
  bool cli_model = false;    // --model given
  std::string config_path;   // --config PATH
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
      app.input_file = argv[++i];
    } else if (std::strcmp(argv[i], "--wake-test") == 0 && i + 1 < argc) {
      app.wake_test_file = argv[++i];
    } else if (std::strcmp(argv[i], "--vad-model") == 0 && i + 1 < argc) {
      app.vad_model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      app.stt_model_path = argv[++i];
      cli_model = true;
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
    } else {
      std::fprintf(stderr,
                   "usage: %s [--input FILE] [--config PATH] "
                   "[--vad-model PATH] [--model PATH] [--wake-test FILE]\n",
                   argv[0]);
      return 2;
    }
  }

  // Suppress whisper/ggml per-frame VAD INFO logging before the model loads
  // (its load-time INFO lines would otherwise reach stderr too).
  whisper_log_set(&WhisperLogSink, nullptr);

  // Configuration. Two plain key=value files, both D12 syntax (a missing
  // file is not an error; an invalid value exits 2):
  //   ~/.config/voice-to-text/dictate.conf       — the legacy hotkey/VAD/
  //     model config, always read at its fixed path.
  //   ~/.config/voice-to-text/autodictate.conf   — the wake phrase and
  //     pause-loop thresholds (ticket 04); --config PATH overrides its
  //     path.
  // Precedence: CLI flags > autodictate.conf > dictate.conf > built-in
  // defaults.
  const char *home = std::getenv("HOME");
  const std::string config_dir = (home && *home)
      ? std::string(home) + "/.config/voice-to-text/"
      : std::string();
  const std::string legacy_path = config_dir + "dictate.conf";
  const std::string auto_path =
      config_path.empty() ? config_dir + "autodictate.conf" : config_path;
  Config config;
  std::string config_error;
  if (!legacy_path.empty() &&
      !LoadConfig(legacy_path, &config, &config_error)) {
    std::fprintf(stderr, "%s\n", config_error.c_str());
    return 2;
  }
  AutoConfig auto_config;
  std::string auto_config_error;
  if (!auto_path.empty() &&
      !LoadAutoConfig(auto_path, &auto_config, &auto_config_error)) {
    std::fprintf(stderr, "%s\n", auto_config_error.c_str());
    return 2;
  }
  app.hold_mods = config.hold_mods != 0
                      ? config.hold_mods
                      : (kModCtrl | kModAlt);
  app.hold_label = config.hold_label.empty() ? "ctrl+alt" : config.hold_label;
  app.vad_threshold = whisper_vad_default_params().threshold;
  if (config.vad_threshold >= 0.0)
    app.vad_threshold = static_cast<float>(config.vad_threshold);
  // Ticket 04: wake phrase + pause-loop thresholds from autodictate.conf
  // (built-in defaults when unset).
  app.wake_phrase = auto_config.wake_phrase.empty()
                        ? kDefaultWakePhrase
                        : auto_config.wake_phrase;
  if (auto_config.pause_chunk_s > 0.0)
    app.pause_chunk_samples =
        static_cast<int64_t>(auto_config.pause_chunk_s *
                             static_cast<double>(kRequestRate));
  if (auto_config.session_end_s > 0.0)
    app.session_end_samples =
        static_cast<int64_t>(auto_config.session_end_s *
                             static_cast<double>(kRequestRate));
  // Model: CLI --model > autodictate.conf model > dictate.conf model >
  // built-in default <exedir>/models/ggml-base.en.bin.
  std::string configured_model;
  if (!auto_config.model.empty())
    configured_model = auto_config.model;
  else if (!config.model.empty())
    configured_model = config.model;
  if (!cli_model && !configured_model.empty())
    app.stt_model_path = ResolveModelPath(configured_model);
  // STT model path for the load below and the startup summary: a CLI
  // --model value is used as-is (existing behavior), the configured value is
  // resolved above, and the default is <exedir>/models/ggml-base.en.bin.
  const std::string stt_path = app.stt_model_path.empty()
      ? ExecutableDirectory() + "/models/ggml-base.en.bin"
      : app.stt_model_path;
  std::printf("dictate: config: hold_key=%s model=%s vad_threshold=%.2f "
              "gpu_device=%s\n",
              app.hold_label.c_str(),
              stt_path.c_str(), app.vad_threshold,
              config.gpu_device.empty() ? "(none)" : config.gpu_device.c_str());
  std::printf("dictate: autodictate: wake_phrase=\"%s\" pause_chunk_s=%.1f "
              "session_end_s=%.1f config=%s\n",
              app.wake_phrase.c_str(),
              static_cast<double>(app.pause_chunk_samples) /
                  static_cast<double>(kRequestRate),
              static_cast<double>(app.session_end_samples) /
                  static_cast<double>(kRequestRate),
              auto_path.c_str());
  if (!config.gpu_device.empty())
    std::printf("dictate: note: gpu_device is set but ignored "
                "(CPU-only build)\n");

  // Load the Silero VAD model (whisper.cpp native ggml build, not ONNX).
  // Default: <executable's own dir>/models/ggml-silero-v6.2.0.bin — resolved
  // from the executable path, not the current working directory.
  const std::string model_path = app.vad_model_path.empty()
      ? ExecutableDirectory() + std::string("/") + "models" + "/" +
            kDefaultVadModelName
      : app.vad_model_path;
  const struct whisper_vad_context_params vad_ctx_params =
      whisper_vad_default_context_params();
  app.vctx =
      whisper_vad_init_from_file_with_params(model_path.c_str(), vad_ctx_params);
  if (!app.vctx) {
    std::fprintf(stderr, "dictate: failed to load VAD model '%s'\n",
                 model_path.c_str());
    return 1;
  }
  // Load the STT model (whisper.cpp, CPU-only). The path is resolved above:
  // a CLI --model value as given, the configured model (resolved under
  // <exedir>/models/), or the default <exedir>/models/ggml-base.en.bin.
  app.stt = whisper_init_from_file_with_params(
      stt_path.c_str(), whisper_context_default_params());
  if (!app.stt) {
    std::fprintf(stderr, "dictate: failed to load STT model '%s'\n",
                 stt_path.c_str());
    return 1;
  }

  if (!app.wake_test_file.empty()) {
    // Wake-test mode (ticket 02): run the idle wake-detection pipeline
    // (VAD gate -> completed-segment transcription -> fuzzy match) against a
    // 16 kHz mono WAV with no real-time pacing. Prints the match decision
    // for every completed speech segment and exits.
    std::vector<int16_t> file_pcm;
    uint32_t rate = 0, channels = 0;
    std::string error;
    if (!ReadWav16(app.wake_test_file.c_str(), &file_pcm, &rate, &channels,
                   &error)) {
      std::fprintf(stderr, "dictate: %s\n", error.c_str());
      return 1;
    }
    if (rate != kRequestRate || channels != kRequestChannels) {
      std::fprintf(stderr,
                   "dictate: input must be %u Hz mono 16-bit PCM, got %u Hz, "
                   "%u ch\n",
                   kRequestRate, rate, channels);
      return 1;
    }
    std::printf("dictate: wake-test: %s (%.3f s), wake phrase: \"%s\"\n",
                app.wake_test_file.c_str(),
                static_cast<double>(file_pcm.size()) / kRequestRate,
                app.wake_phrase.c_str());
    bool matched = false;
    const int64_t total = static_cast<int64_t>(file_pcm.size());
    const int64_t chunk = kRequestRate;  // 1 s per feed
    for (int64_t off = 0; off < total && !matched; off += chunk)
      matched = app.wake.Feed(&app, file_pcm.data() + off,
                               std::min(chunk, total - off));
    bool flush_match = false;
    app.wake.Flush(&app, &flush_match);
    if (flush_match) matched = true;
    if (matched) std::printf("\xF0\x9F\x8E\x99 capture started\n");
    std::printf("wake-test: %d segment(s) checked, last text: \"%s\"\n",
                app.wake.segments, app.wake.last_text.c_str());
    std::printf("wake-test: %s\n", matched ? "MATCH" : "no match");
    return 0;
  }

  if (!app.input_file.empty()) {
    // File mode: read a 16 kHz mono S16 WAV through the identical pipeline.
    std::vector<int16_t> file_pcm;
    uint32_t rate = 0, channels = 0;
    std::string error;
    if (!ReadWav16(app.input_file.c_str(), &file_pcm, &rate, &channels,
                   &error)) {
      std::fprintf(stderr, "dictate: %s\n", error.c_str());
      return 1;
    }
    if (rate != kRequestRate || channels != kRequestChannels) {
      std::fprintf(stderr,
                   "dictate: input must be %u Hz mono 16-bit PCM, got %u Hz, %u ch\n",
                   kRequestRate, rate, channels);
      return 1;
    }
    app.negotiated_rate = rate;
    app.negotiated_channels = channels;
    // Run the session over the file with the identical pause-loop logic as
    // the daemon (ticket 03): a 2 s pause produces a transcribed+typed
    // chunk, 20 s of continuous silence ends the session, and audio still
    // buffered at input end is flushed as a final chunk. One VAD frame at a
    // time so the chunk/end checks run at every frame boundary.
    ResetSession(&app);
    app.session_active = true;
    app.capturing = true;
    const int64_t total = static_cast<int64_t>(file_pcm.size());
    int64_t pos = 0;
    while (pos < total && app.vad_ok && app.stop_reason.empty()) {
      const int64_t n = std::min<int64_t>(kVadFrameSamples, total - pos);
      const int64_t new_captured = app.captured + n;
      if (app.pcm.size() < static_cast<std::size_t>(new_captured))
        app.pcm.resize(static_cast<std::size_t>(new_captured));
      std::memcpy(app.pcm.data() + app.captured, file_pcm.data() + pos,
                  static_cast<std::size_t>(n * 2));
      app.captured = new_captured;
      app.session_total_samples += n;
      app.vad_feed_offset +=
          VADFeed(&app, app.pcm.data() + app.vad_feed_offset,
                  app.captured - app.vad_feed_offset, false);
      pos += n;
      if (app.chunk_requested) {
        app.capturing = false;
        ProcessChunk(&app);
        app.capturing = true;
      }
    }
    if (app.stop_reason.empty()) app.stop_reason = "input end";
    if (!app.vad_ok) {
      std::fprintf(stderr, "dictate: VAD computation failed\n");
      return 1;
    }
    // Flush whatever is still buffered (e.g. trailing speech the file ends
    // on) as a final chunk, then end the session cleanly.
    if (!app.pcm.empty()) {
      app.capturing = false;
      ProcessChunk(&app);
      app.capturing = true;
    }
    app.session_active = false;
    app.capturing = false;
    app.pcm.resize(static_cast<std::size_t>(
        std::min(app.captured, app.vad_samples_fed)));
    return EndSession(&app, false, app.input_file, 0);
  }

  // Default mode: the dictation daemon. Connects to PipeWire
  // once, resolves the source once, then idles for a start key and
  // runs per-session capture + transcription + typing.
  pw_init(&argc, &argv);
  return RunDictationDaemon(&app);
}
