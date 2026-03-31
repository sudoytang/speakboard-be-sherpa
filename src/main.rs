mod asr;
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

use asr::load_recognizer;
use ws_handler::handle_socket;

#[derive(Clone)]
struct AppState {
    recognizer: Arc<Mutex<SenseVoiceRecognizer>>,
}

async fn ws_route(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let recognizer = Arc::clone(&state.recognizer);
    ws.on_upgrade(move |socket| handle_socket(socket, recognizer))
}

fn main() {
    // Default: our crate at debug, noisy dependencies at info.
    // Override with RUST_LOG env var, e.g. RUST_LOG=trace cargo run
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info,speakboard_be_sherpa=debug")
    });
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    let num_threads: i32 = std::env::var("NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());

    // Ensure model files are on disk (download if missing).
    let paths = model_dl::ensure_model_paths(num_threads)
        .expect("Failed to prepare model files");

    // Load the model into memory once, before accepting any connections.
    info!("Loading ASR model...");
    let recognizer = load_recognizer(&paths).expect("Failed to load ASR model");
    info!("ASR model loaded.");

    let state = AppState { recognizer: Arc::new(Mutex::new(recognizer)) };

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime")
        .block_on(serve(state, port));
}

async fn serve(state: AppState, port: String) {
    let app = Router::new()
        .route("/ws", get(ws_route))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("Listening on ws://{addr}/ws");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
