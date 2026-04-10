mod asr;
mod config;
mod model_dl;
mod protocol;
mod session;

use std::sync::{Arc, Mutex};

use sherpa_rs::sense_voice::SenseVoiceRecognizer;
use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::EnvFilter;

use asr::{AsrConfig, load_recognizer};
use config::{ResolvedConfig, TransportKind};
use session::run_session;

#[derive(Clone)]
struct AppState {
    recognizer: Arc<Mutex<SenseVoiceRecognizer>>,
    asr_config: AsrConfig,
}

fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info,speakboard_be_sherpa=debug")
    });
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    let cfg = config::load(config::parse_config_path().as_deref())
        .expect("Failed to load config");

    // Ensure model files are on disk (download if missing).
    let paths = model_dl::ensure_model_paths(
        cfg.num_threads,
        cfg.model_path.clone(),
        cfg.tokens_path.clone(),
    )
    .expect("Failed to prepare model files");

    // Load the model into memory once, before accepting any connections.
    info!("Loading ASR model...");
    let recognizer = load_recognizer(&paths).expect("Failed to load ASR model");
    info!("ASR model loaded.");

    let asr_config = AsrConfig::from_resolved(&cfg);
    let state = AppState {
        recognizer: Arc::new(Mutex::new(recognizer)),
        asr_config,
    };

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime")
        .block_on(serve(state, cfg));
}

async fn serve(state: AppState, cfg: ResolvedConfig) {
    match cfg.transport {
        TransportKind::LoopbackTcp => serve_tcp(state, cfg.port).await,
        TransportKind::UnixDomainSocket => serve_unix(state, &cfg.socket_path).await,
    }
}

async fn serve_tcp(state: AppState, port: u16) {
    let addr = format!("127.0.0.1:{port}");
    info!("Listening on tcp://{addr}");

    let listener = TcpListener::bind(&addr).await.unwrap();
    loop {
        let Ok((stream, peer)) = listener.accept().await else {
            continue;
        };
        info!("Accepted TCP client from {peer}");
        let recognizer = Arc::clone(&state.recognizer);
        let asr_config = state.asr_config.clone();
        tokio::spawn(async move {
            run_session(stream, recognizer, asr_config).await;
        });
    }
}

#[cfg(unix)]
async fn serve_unix(state: AppState, socket_path: &str) {
    use std::path::{Path, PathBuf};

    use tokio::net::UnixListener;

    struct SocketCleanupGuard(PathBuf);

    impl Drop for SocketCleanupGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    let path = Path::new(socket_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }

    info!("Listening on unix://{}", path.display());
    let _cleanup = SocketCleanupGuard(path.to_path_buf());
    let listener = UnixListener::bind(path).unwrap();
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let recognizer = Arc::clone(&state.recognizer);
        let asr_config = state.asr_config.clone();
        tokio::spawn(async move {
            run_session(stream, recognizer, asr_config).await;
        });
    }
}

#[cfg(not(unix))]
async fn serve_unix(_state: AppState, socket_path: &str) {
    panic!("Unix Domain Socket transport is not supported on this platform: {socket_path}");
}
