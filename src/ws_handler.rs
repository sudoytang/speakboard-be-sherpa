use std::sync::{Arc, Mutex};

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use sherpa_rs::sense_voice::SenseVoiceRecognizer;
use tracing::{debug, info, warn};

use crate::asr::{spawn_asr_thread, AsrConfig, AsrResult, AudioMsg};
use crate::protocol::{ClientMessage, ServerMessage};

pub async fn handle_socket(
    socket: WebSocket,
    recognizer: Arc<Mutex<SenseVoiceRecognizer>>,
    asr_config: AsrConfig,
) {
    let (audio_tx, mut result_rx) = spawn_asr_thread(recognizer, asr_config);
    let (mut ws_tx, mut ws_rx) = socket.split();

    match result_rx.recv().await {
        Some(AsrResult::Ready) => {
            let msg = serde_json::to_string(&ServerMessage::Ready).unwrap();
            if ws_tx.send(Message::Text(msg.into())).await.is_err() {
                return;
            }
            info!("session started — sent Ready");
        }
        _ => return,
    }

    // Task A: forward ASR results to the client
    let write_task = tokio::spawn(async move {
        while let Some(result) = result_rx.recv().await {
            let msg = match result {
                AsrResult::Partial { id, text, lang, rms } => {
                    info!("→ WS Partial id={id}  lang={lang}  rms={rms:.4}  text=\"{text}\"");
                    serde_json::to_string(&ServerMessage::Partial { id, text, lang, rms }).unwrap()
                }
                AsrResult::GoldReplace { text, lang, rms } => {
                    info!("→ WS GoldReplace  lang={lang}  rms={rms:.4}  text=\"{text}\"");
                    serde_json::to_string(&ServerMessage::GoldReplace { text, lang, rms }).unwrap()
                }
                AsrResult::Ready => continue,
            };
            if ws_tx.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
        debug!("write_task exiting");
    });

    // Task B: receive audio / control messages from the client
    let read_task = tokio::spawn(async move {
        let mut total_audio_bytes: u64 = 0;
        let mut audio_chunk_count: u64 = 0;
        while let Some(frame) = ws_rx.next().await {
            match frame {
                Ok(Message::Binary(data)) => {
                    total_audio_bytes += data.len() as u64;
                    audio_chunk_count += 1;
                    if audio_chunk_count % 100 == 0 {
                        debug!(
                            "received {audio_chunk_count} audio chunks  \
                             total={:.1}s",
                            total_audio_bytes as f64 / (16_000.0 * 2.0) // i16 = 2 bytes/sample
                        );
                    }
                    let samples = pcm_i16_le_to_f32(&data);
                    if audio_tx.send(AudioMsg::Samples(samples)).await.is_err() {
                        break;
                    }
                }
                Ok(Message::Text(text)) => match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::Stop) => {
                        info!(
                            "Stop received after {audio_chunk_count} chunks ({:.1}s)",
                            total_audio_bytes as f64 / (16_000.0 * 2.0)
                        );
                        let _ = audio_tx.send(AudioMsg::Stop).await;
                        break;
                    }
                    Ok(ClientMessage::Start) => {}
                    Err(e) => warn!("Unknown client message: {e}"),
                },
                Ok(Message::Close(_)) | Err(_) => {
                    let _ = audio_tx.send(AudioMsg::Stop).await;
                    break;
                }
                _ => {}
            }
        }
        debug!("read_task exiting");
    });

    tokio::select! {
        _ = write_task => {}
        _ = read_task  => {}
    }
}

fn pcm_i16_le_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect()
}
