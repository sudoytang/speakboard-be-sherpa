/// CLI test client for the Speakboard ASR backend.
///
/// Usage:
///   cargo run --bin client [-- --debug]
///   cargo run --bin client [-- --tcp 8080]
///   cargo run --bin client [-- --socket /tmp/speakboard-sidecar.sock]
///
/// Controls:
///   SPACE / ENTER  — start recording (connects to server, opens mic)
///   SPACE / ENTER  — stop recording (flushes audio, waits for final results)
///   Q / ESC        — quit

use std::{env, io::Write, time::Duration};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal,
};
use serde::Deserialize;
use tokio::{
    io::{self, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpStream,
    sync::{mpsc, oneshot},
};

#[cfg(unix)]
use tokio::net::UnixStream;

const FRAME_JSON: u8 = 1;
const FRAME_AUDIO: u8 = 2;
const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone)]
enum ClientEndpoint {
    LoopbackTcp(u16),
    #[cfg(unix)]
    UnixSocket(String),
}

impl ClientEndpoint {
    fn from_args() -> Self {
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tcp" => {
                    if let Some(port) = args.next().and_then(|value| value.parse().ok()) {
                        return Self::LoopbackTcp(port);
                    }
                }
                "--socket" => {
                    #[cfg(unix)]
                    if let Some(path) = args.next() {
                        return Self::UnixSocket(path);
                    }
                }
                _ => {}
            }
        }

        #[cfg(unix)]
        {
            Self::UnixSocket(default_socket_path())
        }
        #[cfg(not(unix))]
        {
            Self::LoopbackTcp(8080)
        }
    }

    fn description(&self) -> String {
        match self {
            Self::LoopbackTcp(port) => format!("tcp://127.0.0.1:{port}"),
            #[cfg(unix)]
            Self::UnixSocket(path) => format!("unix://{path}"),
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMsg {
    Ready,
    Partial { text: String, lang: String, rms: f32 },
    GoldReplace { text: String, lang: String, rms: f32 },
    #[serde(other)]
    Unknown,
}

enum TranscriptMsg {
    Partial { text: String, lang: String, rms: f32 },
    Gold { text: String, lang: String, rms: f32 },
}

struct Session {
    _stream: cpal::Stream,
    stop_tx: oneshot::Sender<()>,
}

impl Session {
    fn stop(self) {
        let _ = self.stop_tx.send(());
    }
}

enum ClientStream {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(UnixStream),
}

async fn start_session(
    endpoint: ClientEndpoint,
    transcript_tx: mpsc::Sender<TranscriptMsg>,
) -> Result<Session> {
    match connect_stream(&endpoint).await? {
        ClientStream::Tcp(stream) => start_session_on_stream(stream, transcript_tx).await,
        #[cfg(unix)]
        ClientStream::Unix(stream) => start_session_on_stream(stream, transcript_tx).await,
    }
}

async fn connect_stream(endpoint: &ClientEndpoint) -> Result<ClientStream> {
    match endpoint {
        ClientEndpoint::LoopbackTcp(port) => {
            let stream = TcpStream::connect(("127.0.0.1", *port))
                .await
                .context("Could not connect to loopback TCP server")?;
            Ok(ClientStream::Tcp(stream))
        }
        #[cfg(unix)]
        ClientEndpoint::UnixSocket(path) => {
            let stream = UnixStream::connect(path)
                .await
                .with_context(|| format!("Could not connect to unix socket: {path}"))?;
            Ok(ClientStream::Unix(stream))
        }
    }
}

async fn start_session_on_stream<S>(
    stream: S,
    transcript_tx: mpsc::Sender<TranscriptMsg>,
) -> Result<Session>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (mut reader, mut writer) = io::split(stream);

    loop {
        match read_frame(&mut reader).await? {
            Some((FRAME_JSON, payload)) => {
                let text = String::from_utf8(payload).context("Server sent invalid UTF-8")?;
                if let Ok(ServerMsg::Ready) = serde_json::from_str(&text) {
                    break;
                }
            }
            Some(_) => {}
            None => anyhow::bail!("Server closed connection before sending ready"),
        }
    }

    write_json_frame(&mut writer, br#"{"type":"start"}"#).await?;

    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<u8>>(256);
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;

                _ = &mut stop_rx => {
                    while let Ok(bytes) = audio_rx.try_recv() {
                        if write_frame(&mut writer, FRAME_AUDIO, &bytes).await.is_err() {
                            return;
                        }
                    }
                    let _ = write_json_frame(&mut writer, br#"{"type":"stop"}"#).await;
                    while let Ok(Some((FRAME_JSON, payload))) = read_frame(&mut reader).await {
                        if let Ok(text) = String::from_utf8(payload) {
                            forward_msg(&text, &transcript_tx).await;
                        }
                    }
                    return;
                }

                Some(bytes) = audio_rx.recv() => {
                    if write_frame(&mut writer, FRAME_AUDIO, &bytes).await.is_err() {
                        return;
                    }
                }

                frame = read_frame(&mut reader) => {
                    match frame {
                        Ok(Some((FRAME_JSON, payload))) => {
                            if let Ok(text) = String::from_utf8(payload) {
                                forward_msg(&text, &transcript_tx).await;
                            }
                        }
                        Ok(None) | Err(_) => return,
                        _ => {}
                    }
                }
            }
        }
    });

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No audio input device found")?;

    let default_cfg = device
        .default_input_config()
        .context("Failed to get default input config")?;
    let native_rate = default_cfg.sample_rate().0;
    let native_channels = default_cfg.channels() as usize;
    let stream_config: cpal::StreamConfig = default_cfg.into();

    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = if native_channels == 1 {
                    data.to_vec()
                } else {
                    data.chunks(native_channels)
                        .map(|frame| frame.iter().sum::<f32>() / native_channels as f32)
                        .collect()
                };
                let resampled = resample_linear(&mono, native_rate, TARGET_SAMPLE_RATE);
                let bytes: Vec<u8> = resampled
                    .iter()
                    .flat_map(|&s| {
                        let s16 = (s * 32_767.0).clamp(-32_768.0, 32_767.0) as i16;
                        s16.to_le_bytes()
                    })
                    .collect();
                let _ = audio_tx.try_send(bytes);
            },
            |err| eprintln!("\n[audio error] {err}"),
            None,
        )
        .context("Failed to open microphone")?;

    stream.play().context("Failed to start microphone stream")?;

    Ok(Session {
        _stream: stream,
        stop_tx,
    })
}

async fn forward_msg(text: &str, tx: &mpsc::Sender<TranscriptMsg>) {
    match serde_json::from_str::<ServerMsg>(text) {
        Ok(ServerMsg::Partial { text, lang, rms }) => {
            tx.send(TranscriptMsg::Partial { text, lang, rms })
                .await
                .ok();
        }
        Ok(ServerMsg::GoldReplace { text, lang, rms }) => {
            tx.send(TranscriptMsg::Gold { text, lang, rms }).await.ok();
        }
        _ => {}
    }
}

// ── Display helpers ──────────────────────────────────────────────────────────

/// Format the debug annotation shown after a partial or gold line.
fn debug_tag(lang: &str, rms: f32) -> String {
    format!("\x1b[2m [lang={lang} rms={rms:.3}]\x1b[0m")
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let endpoint = ClientEndpoint::from_args();
    let debug_mode = std::env::args().any(|a| a == "--debug");

    let (transcript_tx, mut transcript_rx) = mpsc::channel::<TranscriptMsg>(32);

    let (key_tx, mut key_rx) = mpsc::channel::<KeyCode>(16);
    tokio::task::spawn_blocking(move || {
        terminal::enable_raw_mode().expect("enable raw mode");
        loop {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => {
                    if let Ok(Event::Key(k)) = event::read() {
                        if k.kind == KeyEventKind::Press
                            && key_tx.blocking_send(k.code).is_err()
                        {
                            break;
                        }
                    }
                }
                Ok(false) => {}
                Err(_) => break,
            }
        }
        terminal::disable_raw_mode().ok();
    });

    println!("╭────────────────────────────────────╮");
    println!("│   Speakboard ASR — test client     │");
    println!("│                                    │");
    println!("│  SPACE / ENTER  start / stop       │");
    println!("│  Q / ESC        quit               │");
    if debug_mode {
        println!("│  mode: DEBUG                       │");
    }
    println!("╰────────────────────────────────────╯");
    println!();
    println!("  → Press SPACE to start recording");

    let mut session: Option<Session> = None;

    // Accumulate pending partial entries (text + metadata) for display.
    let mut partials: Vec<(String, String, f32)> = Vec::new(); // (text, lang, rms)

    loop {
        tokio::select! {
            Some(key) = key_rx.recv() => {
                match key {
                    KeyCode::Char(' ') | KeyCode::Enter => {
                        if session.is_none() {
                            print!("  Connecting to {} ...", endpoint.description());
                            std::io::stdout().flush().ok();
                            match start_session(endpoint.clone(), transcript_tx.clone()).await {
                                Ok(s) => {
                                    session = Some(s);
                                    partials.clear();
                                    print!("\r\x1b[2K  🎙  Recording — press SPACE to stop\r\n");
                                }
                                Err(e) => {
                                    print!("\r\x1b[2K  [error] {e:#}\r\n");
                                }
                            }
                        } else {
                            print!("\r\x1b[2K  ⏹  Finalizing...\r\n");
                            session.take().unwrap().stop();
                            print!("\r\x1b[2K  → Press SPACE to record again\r\n");
                        }
                    }
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                        if let Some(s) = session.take() {
                            print!("\r\x1b[2K  Stopping...\r\n");
                            s.stop();
                            tokio::time::sleep(Duration::from_millis(500)).await;
                        }
                        break;
                    }
                    _ => {}
                }
            }

            Some(msg) = transcript_rx.recv() => {
                match msg {
                    TranscriptMsg::Partial { text, lang, rms } => {
                        partials.push((text, lang, rms));
                        // Previous partials in gray, newest in green.
                        let mut preview = String::new();
                        for (i, (p, p_lang, p_rms)) in partials.iter().enumerate() {
                            if i > 0 { preview.push(' '); }
                            if i < partials.len() - 1 {
                                preview.push_str(&format!("\x1b[90m{p}\x1b[0m"));
                                if debug_mode {
                                    preview.push_str(&format!("\x1b[90m{}\x1b[0m", debug_tag(p_lang, *p_rms)));
                                }
                            } else {
                                preview.push_str(&format!("\x1b[32m{p}\x1b[0m"));
                                if debug_mode {
                                    preview.push_str(&format!("\x1b[32m{}\x1b[0m", debug_tag(p_lang, *p_rms)));
                                }
                            }
                        }
                        print!("\r\x1b[2K  ✏️  [{preview}]\r\n");
                    }
                    TranscriptMsg::Gold { text, lang, rms } => {
                        partials.clear();
                        let tag = if debug_mode { debug_tag(&lang, rms) } else { String::new() };
                        print!("\r\x1b[2K  📝  {text}{tag}\r\n");
                    }
                }
                std::io::stdout().flush().ok();
            }
        }
    }

    print!("\r\x1b[2KBye.\r\n");
    Ok(())
}

async fn read_frame<R>(reader: &mut R) -> io::Result<Option<(u8, Vec<u8>)>>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0_u8; 5];
    match reader.read_exact(&mut header).await {
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }

    let len = u32::from_be_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let mut payload = vec![0_u8; len];
    reader.read_exact(&mut payload).await?;
    Ok(Some((header[0], payload)))
}

async fn write_json_frame<W>(writer: &mut W, payload: &[u8]) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_frame(writer, FRAME_JSON, payload).await
}

async fn write_frame<W>(writer: &mut W, frame_type: u8, payload: &[u8]) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    writer.write_u8(frame_type).await?;
    writer.write_u32(payload.len() as u32).await?;
    writer.write_all(payload).await?;
    writer.flush().await
}

#[cfg(unix)]
fn default_socket_path() -> String {
    std::env::temp_dir()
        .join("speakboard-sidecar.sock")
        .display()
        .to_string()
}

/// Resample `samples` from `from_hz` to `to_hz` using linear interpolation.
fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let s0 = samples[idx];
            let s1 = samples.get(idx + 1).copied().unwrap_or(s0);
            s0 + (s1 - s0) * frac
        })
        .collect()
}
