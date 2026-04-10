use serde::Deserialize;

use anyhow::{Context, Result};
use tracing::info;

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TransportKind {
    #[default]
    UnixDomainSocket,
    LoopbackTcp,
}

/// JSON schema for the on-disk config file.
/// Every field is optional; missing fields fall back to the built-in defaults.
///
/// Example file:
/// ```json
/// {
///   "transport": "unix_domain_socket",
///   "socket_path": "/var/folders/.../speakboard-sidecar.sock",
///   "port": 8080,
///   "num_threads": 4,
///   "silence_rms_threshold": 0.02,
///   "partial_silence_secs": 0.8,
///   "gold_silence_secs": 2.0,
///   "max_gold_secs": 30.0,
///   "min_transcribe_secs": 0.5,
///   "min_speech_secs": 0.3
/// }
/// ```
#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct ConfigFile {
    pub transport: Option<TransportKind>,
    pub port: Option<u16>,
    pub socket_path: Option<String>,
    pub num_threads: Option<i32>,
    /// Override model .onnx path (skips auto-download when set).
    pub model_path: Option<String>,
    /// Override tokens.txt path (skips auto-download when set).
    pub tokens_path: Option<String>,

    pub silence_rms_threshold: Option<f32>,
    pub partial_silence_secs: Option<f64>,
    pub gold_silence_secs: Option<f64>,
    pub max_gold_secs: Option<f64>,
    pub min_transcribe_secs: Option<f64>,
    pub min_speech_secs: Option<f64>,
}

/// Fully resolved configuration with concrete values.
/// Built from a ConfigFile merged with env-var overrides and built-in defaults.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub transport: TransportKind,
    pub port: u16,
    pub socket_path: String,
    pub num_threads: i32,
    pub model_path: Option<String>,
    pub tokens_path: Option<String>,

    pub silence_rms_threshold: f32,
    pub partial_silence_secs: f64,
    pub gold_silence_secs: f64,
    pub max_gold_secs: f64,
    pub min_transcribe_secs: f64,
    pub min_speech_secs: f64,
}

impl Default for ResolvedConfig {
    fn default() -> Self {
        Self {
            transport: TransportKind::UnixDomainSocket,
            port: 8080,
            socket_path: default_socket_path(),
            num_threads: 4,
            model_path: None,
            tokens_path: None,
            silence_rms_threshold: 0.02,
            partial_silence_secs: 0.8,
            gold_silence_secs: 2.0,
            max_gold_secs: 30.0,
            min_transcribe_secs: 0.5,
            min_speech_secs: 0.3,
        }
    }
}

/// Load configuration from an optional JSON file path.
///
/// Resolution priority (highest → lowest):
///   1. Environment variables (`PORT`, `NUM_THREADS`, `MODEL_PATH`, `TOKENS_PATH`)
///   2. JSON config file fields
///   3. Built-in defaults
pub fn load(path: Option<&str>) -> Result<ResolvedConfig> {
    let file: ConfigFile = match path {
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .with_context(|| format!("Cannot read config file: {p}"))?;
            serde_json::from_str(&text)
                .with_context(|| format!("Invalid JSON in config file: {p}"))?
        }
        None => ConfigFile::default(),
    };

    let defaults = ResolvedConfig::default();

    let resolved = ResolvedConfig {
        transport: env_transport("TRANSPORT")
            .unwrap_or(file.transport.unwrap_or(defaults.transport)),
        port: env_u16("PORT").unwrap_or(file.port.unwrap_or(defaults.port)),
        socket_path: std::env::var("SOCKET_PATH")
            .ok()
            .or(file.socket_path)
            .unwrap_or(defaults.socket_path),
        num_threads: env_i32("NUM_THREADS")
            .unwrap_or(file.num_threads.unwrap_or(defaults.num_threads)),
        model_path: std::env::var("MODEL_PATH")
            .ok()
            .or(file.model_path),
        tokens_path: std::env::var("TOKENS_PATH")
            .ok()
            .or(file.tokens_path),
        silence_rms_threshold: file
            .silence_rms_threshold
            .unwrap_or(defaults.silence_rms_threshold),
        partial_silence_secs: file
            .partial_silence_secs
            .unwrap_or(defaults.partial_silence_secs),
        gold_silence_secs: file
            .gold_silence_secs
            .unwrap_or(defaults.gold_silence_secs),
        max_gold_secs: file.max_gold_secs.unwrap_or(defaults.max_gold_secs),
        min_transcribe_secs: file
            .min_transcribe_secs
            .unwrap_or(defaults.min_transcribe_secs),
        min_speech_secs: file.min_speech_secs.unwrap_or(defaults.min_speech_secs),
    };

    let endpoint = match resolved.transport {
        TransportKind::UnixDomainSocket => format!("unix://{}", resolved.socket_path),
        TransportKind::LoopbackTcp => format!("tcp://127.0.0.1:{}", resolved.port),
    };
    info!(
        "Config: endpoint={} threads={} rms_thresh={:.3} \
         partial={:.1}s gold={:.1}s max={:.0}s \
         min_transcribe={:.2}s min_speech={:.2}s",
        endpoint,
        resolved.num_threads,
        resolved.silence_rms_threshold,
        resolved.partial_silence_secs,
        resolved.gold_silence_secs,
        resolved.max_gold_secs,
        resolved.min_transcribe_secs,
        resolved.min_speech_secs,
    );

    Ok(resolved)
}

/// Parse `--config <path>` from process arguments.
pub fn parse_config_path() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let pos = args.iter().position(|a| a == "--config")?;
    args.get(pos + 1).cloned()
}

fn env_u16(key: &str) -> Option<u16> {
    std::env::var(key).ok()?.parse().ok()
}

fn env_i32(key: &str) -> Option<i32> {
    std::env::var(key).ok()?.parse().ok()
}

fn env_transport(key: &str) -> Option<TransportKind> {
    match std::env::var(key).ok()?.as_str() {
        "unix_domain_socket" => Some(TransportKind::UnixDomainSocket),
        "loopback_tcp" => Some(TransportKind::LoopbackTcp),
        _ => None,
    }
}

fn default_socket_path() -> String {
    std::env::temp_dir()
        .join("speakboard-sidecar.sock")
        .display()
        .to_string()
}
