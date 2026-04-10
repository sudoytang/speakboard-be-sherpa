use std::io::{self, ErrorKind};
use std::sync::{Arc, Mutex};

use sherpa_rs::sense_voice::SenseVoiceRecognizer;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tracing::{debug, info, warn};

use crate::asr::{AsrConfig, AsrResult, AudioMsg, spawn_asr_thread};
use crate::protocol::{ClientMessage, ServerMessage};

const FRAME_JSON: u8 = 1;
const FRAME_AUDIO: u8 = 2;

pub async fn run_session<S>(
    stream: S,
    recognizer: Arc<Mutex<SenseVoiceRecognizer>>,
    asr_config: AsrConfig,
) where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (audio_tx, mut result_rx) = spawn_asr_thread(recognizer, asr_config);
    let (mut reader, mut writer) = tokio::io::split(stream);

    if write_json_frame(&mut writer, &ServerMessage::Ready)
        .await
        .is_err()
    {
        return;
    }
    info!("session started — sent Ready");

    let write_task = tokio::spawn(async move {
        while let Some(result) = result_rx.recv().await {
            let msg = match result {
                AsrResult::Partial {
                    id,
                    text,
                    lang,
                    rms,
                } => {
                    info!("→ Partial id={id}  lang={lang}  rms={rms:.4}  text=\"{text}\"");
                    ServerMessage::Partial {
                        id,
                        text,
                        lang,
                        rms,
                    }
                }
                AsrResult::GoldReplace { text, lang, rms } => {
                    info!("→ GoldReplace  lang={lang}  rms={rms:.4}  text=\"{text}\"");
                    ServerMessage::GoldReplace { text, lang, rms }
                }
                AsrResult::Ready => continue,
            };

            if write_json_frame(&mut writer, &msg).await.is_err() {
                break;
            }
        }
        debug!("write_task exiting");
    });

    let read_task = tokio::spawn(async move {
        let mut total_audio_bytes: u64 = 0;
        let mut audio_chunk_count: u64 = 0;

        loop {
            match read_frame(&mut reader).await {
                Ok(Some((FRAME_AUDIO, data))) => {
                    total_audio_bytes += data.len() as u64;
                    audio_chunk_count += 1;
                    if audio_chunk_count % 100 == 0 {
                        debug!(
                            "received {audio_chunk_count} audio chunks  total={:.1}s",
                            total_audio_bytes as f64 / (16_000.0 * 2.0)
                        );
                    }
                    let samples = pcm_i16_le_to_f32(&data);
                    if audio_tx.send(AudioMsg::Samples(samples)).await.is_err() {
                        break;
                    }
                }
                Ok(Some((FRAME_JSON, data))) => {
                    let Ok(text) = String::from_utf8(data) else {
                        warn!("Ignoring non-UTF8 json frame");
                        continue;
                    };
                    match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::Stop) => {
                            info!(
                                "Stop received after {audio_chunk_count} chunks ({:.1}s)",
                                total_audio_bytes as f64 / (16_000.0 * 2.0)
                            );
                            let _ = audio_tx.send(AudioMsg::Stop).await;
                            break;
                        }
                        Ok(ClientMessage::Start) => {}
                        Err(err) => warn!("Unknown client message: {err}"),
                    }
                }
                Ok(Some((kind, _))) => {
                    warn!("Ignoring unknown frame type: {kind}");
                }
                Ok(None) => {
                    let _ = audio_tx.send(AudioMsg::Stop).await;
                    break;
                }
                Err(err) => {
                    warn!("session read error: {err}");
                    let _ = audio_tx.send(AudioMsg::Stop).await;
                    break;
                }
            }
        }
        debug!("read_task exiting");
    });

    tokio::select! {
        _ = write_task => {}
        _ = read_task  => {}
    }
}

async fn read_frame<R>(reader: &mut R) -> io::Result<Option<(u8, Vec<u8>)>>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0_u8; 5];
    match reader.read_exact(&mut header).await {
        Ok(_) => {}
        Err(err) if err.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }

    let len = u32::from_be_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let mut payload = vec![0_u8; len];
    reader.read_exact(&mut payload).await?;
    Ok(Some((header[0], payload)))
}

async fn write_json_frame<W>(writer: &mut W, msg: &ServerMessage) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let payload = serde_json::to_vec(msg)
        .map_err(|err| io::Error::new(ErrorKind::InvalidData, err))?;
    write_frame(writer, FRAME_JSON, &payload).await
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

fn pcm_i16_le_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect()
}
