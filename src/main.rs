mod asr;
mod config;
mod model_dl;
mod protocol;
mod ws_handler;

use std::sync::{Arc, Mutex};

use axum::{
    Router,
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use sherpa_rs::sense_voice::SenseVoiceRecognizer;
use tracing::info;
use tracing_subscriber::EnvFilter;

use asr::{AsrConfig, load_recognizer};
use ws_handler::handle_socket;

#[derive(Clone)]
struct AppState {
    recognizer: Arc<Mutex<SenseVoiceRecognizer>>,
    asr_config: AsrConfig,
}

async fn ws_route(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let recognizer = Arc::clone(&state.recognizer);
    let asr_config = state.asr_config.clone();
    ws.on_upgrade(move |socket| handle_socket(socket, recognizer, asr_config))
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

    let port = cfg.port;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime")
        .block_on(serve(state, port));
}

async fn serve(state: AppState, port: u16) {
    let app = Router::new()
        .route("/ws", get(ws_route))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("Listening on ws://{addr}/ws");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
