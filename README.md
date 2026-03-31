# speakboard-be-sherpa

A real-time Automatic Speech Recognition (ASR) server written in Rust, built on top of [sherpa-rs](https://github.com/thewh1teagle/sherpa-rs) and the [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) model. Designed as the backend for the Speakboard app.

## Features

- **WebSocket streaming** — clients stream raw PCM audio; the server streams back transcription results in real time
- **Two-tier transcription**
  - **Partial** (0.8 s silence): fast, low-latency draft result
  - **GoldReplace** (2.0 s silence): accurate re-transcription of the full utterance window, replacing all pending partials
- **Noise gate** — utterances shorter than 0.3 s of real speech are discarded to avoid hallucinations on background noise
- **Auto model download** — on first run the SenseVoice INT8 model (~100 MB) is downloaded and extracted automatically
- **Eager model loading** — the model is loaded once at startup and shared across all connections via `Arc<Mutex<_>>`
- **Force-split** — if a continuous speech segment exceeds 30 s, it is split at the best-scored silence point (score = silence duration × proximity to temporal center) and flushed
- **Multi-language** — SenseVoice supports Chinese, English, Japanese, Korean, and Cantonese with automatic language detection and Inverse Text Normalization (ITN)

## Architecture

```
Client (mic) ──PCM i16 LE──▶ WebSocket ──▶ ws_handler
                                                │
                                         AudioMsg channel
                                                │
                                           run_asr (OS thread)
                                           ┌────────────────┐
                                           │  VAD + state   │
                                           │  machine       │
                                           │                │
                                           │  0.8s silence ─┼──▶ Partial
                                           │  2.0s silence ─┼──▶ GoldReplace
                                           │  30s cap ──────┼──▶ GoldReplace
                                           └────────────────┘
                                                │
                                         AsrResult channel
                                                │
                                         ws_handler ──JSON──▶ WebSocket ──▶ Client
```

## Protocol

All messages are JSON with a `type` discriminant.

### Client → Server

| `type` | Description |
|--------|-------------|
| `start` | Optional; signals intent to stream |
| `stop`  | Flush remaining audio and close |

Audio is sent as raw **binary frames** containing 16-bit signed little-endian PCM at **16 kHz mono**.

### Server → Client

| `type` | Fields | Description |
|--------|--------|-------------|
| `ready` | — | Model loaded, session ready |
| `partial` | `id`, `text`, `lang`, `rms` | Quick draft; may be superseded |
| `gold_replace` | `text`, `lang`, `rms` | Accurate result; replace all pending partials |

`lang` is the detected language code (e.g. `"zh"`, `"en"`). `rms` is the RMS energy of the speech portion of the utterance.

## Getting Started

### Prerequisites

- Rust 1.75+ (`rustup` recommended)
- macOS or Linux (Windows untested)

### Run the server

```bash
cargo run --release
```

On first launch the SenseVoice model is downloaded into `./models/` automatically. The server listens on `ws://0.0.0.0:8080/ws`.

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | TCP port to listen on |
| `NUM_THREADS` | `4` | ONNX Runtime inference threads |
| `RUST_LOG` | `info,speakboard_be_sherpa=debug` | Log level |

### Run the test client

```bash
cargo run --bin client
# or with debug overlay (shows lang + rms on each result):
cargo run --bin client -- --debug
```

Controls:

| Key | Action |
|-----|--------|
| `Space` / `Enter` | Start / stop recording |
| `Q` / `Esc` | Quit |

The client records from the default microphone, resamples to 16 kHz, and streams to the server. Results are displayed as:

- `✏️  [partial text]` — provisional, may change (previous partials in gray, newest in green)
- `📝  gold text` — final accurate transcription

## Project Structure

```
src/
├── main.rs          # Server entry point, AppState, routing
├── asr.rs           # VAD state machine, model interface, two-tier logic
├── ws_handler.rs    # Per-connection WebSocket handler
├── protocol.rs      # Shared message types (ClientMessage, ServerMessage)
├── model_dl.rs      # Auto-download and extraction of model files
└── bin/
    └── client.rs    # CLI test client
```

## License

MIT
