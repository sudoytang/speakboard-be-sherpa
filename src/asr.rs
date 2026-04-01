use std::sync::{Arc, Mutex};

use sherpa_rs::sense_voice::SenseVoiceRecognizer;
use tokio::sync::mpsc;
use tracing::{debug, info, trace};

/// The model always expects 16 kHz mono audio.
const SAMPLE_RATE: u32 = 16_000;

/// Runtime-tunable ASR parameters, derived from [`crate::config::ResolvedConfig`].
/// All time values are pre-converted to sample counts for hot-path efficiency.
#[derive(Debug, Clone)]
pub struct AsrConfig {
    pub silence_rms_threshold: f32,
    /// 0.8 s → quick partial result.
    pub partial_silence_samples: usize,
    /// 2.0 s → accurate gold result.
    pub gold_silence_samples: usize,
    /// 30 s hard cap on the gold accumulation buffer.
    pub max_gold_samples: usize,
    /// Minimum audio sent to the model (avoids ONNX shape errors).
    pub min_transcribe_samples: usize,
    /// Minimum real speech in an utterance before attempting transcription.
    pub min_speech_samples: usize,
}

impl AsrConfig {
    pub fn from_resolved(c: &crate::config::ResolvedConfig) -> Self {
        let sr = SAMPLE_RATE as f64;
        Self {
            silence_rms_threshold: c.silence_rms_threshold,
            partial_silence_samples: (sr * c.partial_silence_secs) as usize,
            gold_silence_samples: (sr * c.gold_silence_secs) as usize,
            max_gold_samples: (sr * c.max_gold_secs) as usize,
            min_transcribe_samples: (sr * c.min_transcribe_secs) as usize,
            min_speech_samples: (sr * c.min_speech_secs) as usize,
        }
    }
}

pub enum AudioMsg {
    Samples(Vec<f32>),
    Stop,
}

#[derive(Debug)]
pub enum AsrResult {
    Ready,
    /// Quick transcription — may be replaced by a subsequent GoldReplace.
    Partial { id: String, text: String, lang: String, rms: f32 },
    /// Accurate transcription covering all audio since the last gold boundary.
    GoldReplace { text: String, lang: String, rms: f32 },
}

/// Structured return value from the model.
struct TranscribeOutput {
    text: String,
    lang: String,
}

pub struct ModelPaths {
    pub model: String,
    pub tokens: String,
    pub num_threads: i32,
}

pub fn load_recognizer(paths: &ModelPaths) -> anyhow::Result<SenseVoiceRecognizer> {
    let config = sherpa_rs::sense_voice::SenseVoiceConfig {
        model: paths.model.clone(),
        tokens: paths.tokens.clone(),
        num_threads: Some(paths.num_threads),
        provider: Some("cpu".to_string()),
        language: "auto".to_string(),
        use_itn: true,
        debug: false,
    };
    SenseVoiceRecognizer::new(config).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Spawn a dedicated OS thread for the ASR pipeline and return its channels.
pub fn spawn_asr_thread(
    recognizer: Arc<Mutex<SenseVoiceRecognizer>>,
    config: AsrConfig,
) -> (mpsc::Sender<AudioMsg>, mpsc::Receiver<AsrResult>) {
    let (audio_tx, audio_rx) = mpsc::channel::<AudioMsg>(128);
    let (result_tx, result_rx) = mpsc::channel::<AsrResult>(32);
    std::thread::spawn(move || run_asr(recognizer, audio_rx, result_tx, config));
    (audio_tx, result_rx)
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn transcribe(
    recognizer: &Arc<Mutex<SenseVoiceRecognizer>>,
    audio: &[f32],
    min_transcribe_samples: usize,
) -> Option<TranscribeOutput> {
    let dur_s = audio.len() as f32 / SAMPLE_RATE as f32;
    if audio.len() < min_transcribe_samples {
        debug!("transcribe: skipped (too short: {dur_s:.2}s)");
        return None;
    }
    debug!("transcribe: calling model on {dur_s:.2}s of audio …");
    let mut r = recognizer.lock().unwrap();
    let result = r.transcribe(SAMPLE_RATE, audio);
    let text = result.text.trim().to_string();
    let lang = result.lang.trim().to_string();
    if text.is_empty() {
        debug!("transcribe: model returned empty string");
        None
    } else {
        info!("transcribe: lang={lang}  \"{}\"", text);
        Some(TranscribeOutput { text, lang })
    }
}

/// Choose the best split point within `gold_buffer` for a forced flush.
///
/// Score = silence_duration × center_proximity
///
/// where center_proximity = 1 − 2·|midpoint/total − 0.5|  (1.0 at center, 0.0 at edges)
///
/// Falls back to the temporal midpoint if no silence regions are recorded.
fn best_split(silence_regions: &[(usize, usize)], total: usize) -> usize {
    silence_regions
        .iter()
        .map(|&(s, e)| {
            let mid = (s + e) as f64 / 2.0;
            let proximity = 1.0 - ((mid / total as f64) - 0.5).abs() * 2.0;
            let score = proximity * (e - s) as f64;
            (score, (s + e) / 2)
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, p)| p)
        .unwrap_or(total / 2)
}

fn run_asr(
    recognizer: Arc<Mutex<SenseVoiceRecognizer>>,
    mut audio_rx: mpsc::Receiver<AudioMsg>,
    result_tx: mpsc::Sender<AsrResult>,
    cfg: AsrConfig,
) {
    let _ = result_tx.blocking_send(AsrResult::Ready);

    // All audio accumulated since the last gold boundary.
    let mut gold_buffer: Vec<f32> = Vec::new();

    // Detected silence intervals (sample indices within gold_buffer).
    let mut silence_regions: Vec<(usize, usize)> = Vec::new();

    // Start of the currently open (not yet closed) silence region.
    let mut open_silence: Option<usize> = None;

    // Index within gold_buffer where the current utterance started.
    let mut utterance_start: usize = 0;

    // Consecutive silence samples counted since the last speech chunk.
    let mut silence_samples: usize = 0;

    let mut in_speech = false;
    // True once a partial has been fired for the current silence run.
    let mut partial_fired = false;
    let mut partial_seq: u64 = 0;
    // Actual speech samples (above threshold) in the current utterance.
    // Used both as a noise gate and to compute the utterance RMS.
    let mut speech_samples_in_utterance: usize = 0;
    // Sum of squared samples for speech chunks — used to compute RMS.
    let mut speech_energy_sum: f64 = 0.0;

    info!(
        "ASR thread started (partial={:.1}s  gold={:.1}s  cap={:.0}s)",
        cfg.partial_silence_samples as f32 / SAMPLE_RATE as f32,
        cfg.gold_silence_samples as f32 / SAMPLE_RATE as f32,
        cfg.max_gold_samples as f32 / SAMPLE_RATE as f32,
    );

    let mut chunk_count: u64 = 0;

    loop {
        match audio_rx.blocking_recv() {
            Some(AudioMsg::Samples(samples)) => {
                let energy = rms(&samples);
                let is_speech = energy > cfg.silence_rms_threshold;
                chunk_count += 1;

                // Print a status line every 50 chunks (~1.6 s at typical rates).
                if chunk_count % 50 == 0 {
                    let gold_s = gold_buffer.len() as f32 / SAMPLE_RATE as f32;
                    let sil_s = silence_samples as f32 / SAMPLE_RATE as f32;
                    debug!(
                        "chunk#{chunk_count}  rms={energy:.4}  in_speech={in_speech}  \
                         silence={sil_s:.2}s  gold_buf={gold_s:.2}s  partial_fired={partial_fired}"
                    );
                }

                trace!(
                    "chunk  rms={energy:.4}  is_speech={is_speech}  \
                     silence_s={:.2}  gold_s={:.2}",
                    silence_samples as f32 / SAMPLE_RATE as f32,
                    gold_buffer.len() as f32 / SAMPLE_RATE as f32,
                );

                if is_speech {
                    // ── entering / continuing speech ──────────────────────────
                    if let Some(start) = open_silence.take() {
                        silence_regions.push((start, gold_buffer.len()));
                    }
                    if !in_speech {
                        utterance_start = gold_buffer.len();
                        speech_samples_in_utterance = 0;
                        speech_energy_sum = 0.0;
                        debug!(
                            "speech started  utterance_start={}  gold_buf={:.2}s",
                            utterance_start,
                            gold_buffer.len() as f32 / SAMPLE_RATE as f32,
                        );
                    }
                    speech_samples_in_utterance += samples.len();
                    speech_energy_sum +=
                        samples.iter().map(|&s| s as f64 * s as f64).sum::<f64>();
                    gold_buffer.extend_from_slice(&samples);
                    in_speech = true;
                    silence_samples = 0;
                    partial_fired = false;

                    // Hard cap: force-flush if gold_buffer exceeds the cap.
                    if gold_buffer.len() >= cfg.max_gold_samples {
                        let split = best_split(&silence_regions, gold_buffer.len());
                        info!("force-split at {:.2}s", split as f32 / SAMPLE_RATE as f32);
                        let right = gold_buffer[split..].to_vec();

                        if let Some(out) = transcribe(&recognizer, &gold_buffer[..split], cfg.min_transcribe_samples) {
                            let split_rms = (speech_energy_sum
                                / speech_samples_in_utterance.max(1) as f64)
                                .sqrt() as f32;
                            let _ = result_tx.blocking_send(AsrResult::GoldReplace {
                                text: out.text,
                                lang: out.lang,
                                rms: split_rms,
                            });
                        }

                        silence_regions = silence_regions
                            .iter()
                            .filter(|&&(_, e)| e > split)
                            .map(|&(s, e)| (s.saturating_sub(split), e - split))
                            .collect();
                        if let Some(ref mut start) = open_silence {
                            *start = start.saturating_sub(split);
                        }
                        utterance_start = utterance_start.saturating_sub(split);
                        gold_buffer = right;
                    }
                } else if in_speech || partial_fired {
                    // ── silence after speech (or after partial was sent) ──────
                    if open_silence.is_none() {
                        open_silence = Some(gold_buffer.len());
                        debug!(
                            "silence started  gold_buf={:.2}s",
                            gold_buffer.len() as f32 / SAMPLE_RATE as f32,
                        );
                    }
                    gold_buffer.extend_from_slice(&samples);
                    silence_samples += samples.len();

                    // partial threshold
                    if silence_samples >= cfg.partial_silence_samples && !partial_fired {
                        let sil_s = silence_samples as f32 / SAMPLE_RATE as f32;
                        let utt_s = (gold_buffer.len() - utterance_start) as f32 / SAMPLE_RATE as f32;
                        let speech_s = speech_samples_in_utterance as f32 / SAMPLE_RATE as f32;
                        info!("PARTIAL threshold hit  silence={sil_s:.2}s  utterance={utt_s:.2}s  speech={speech_s:.2}s");
                        if speech_samples_in_utterance >= cfg.min_speech_samples {
                            let utterance_rms = (speech_energy_sum
                                / speech_samples_in_utterance as f64)
                                .sqrt() as f32;
                            let utterance_audio = &gold_buffer[utterance_start..];
                            if let Some(out) = transcribe(&recognizer, utterance_audio, cfg.min_transcribe_samples) {
                                let id = format!("p{partial_seq}");
                                partial_seq += 1;
                                info!(
                                    "→ sending Partial id={id}  lang={}  rms={utterance_rms:.4}  text=\"{}\"",
                                    out.lang, out.text
                                );
                                let _ = result_tx.blocking_send(AsrResult::Partial {
                                    id,
                                    text: out.text,
                                    lang: out.lang,
                                    rms: utterance_rms,
                                });
                            }
                        } else {
                            debug!("PARTIAL skipped: speech too short ({speech_s:.2}s), likely noise");
                        }
                        in_speech = false;
                        partial_fired = true;
                    }

                    // gold threshold
                    if silence_samples >= cfg.gold_silence_samples {
                        let sil_s = silence_samples as f32 / SAMPLE_RATE as f32;
                        let gold_s = gold_buffer.len() as f32 / SAMPLE_RATE as f32;
                        let speech_s = speech_samples_in_utterance as f32 / SAMPLE_RATE as f32;
                        info!("GOLD threshold hit  silence={sil_s:.2}s  gold_buf={gold_s:.2}s  speech={speech_s:.2}s");
                        if let Some(start) = open_silence.take() {
                            silence_regions.push((start, gold_buffer.len()));
                        }
                        if speech_samples_in_utterance >= cfg.min_speech_samples {
                            let utterance_rms = (speech_energy_sum
                                / speech_samples_in_utterance as f64)
                                .sqrt() as f32;
                            if let Some(out) = transcribe(&recognizer, &gold_buffer, cfg.min_transcribe_samples) {
                                info!(
                                    "→ sending GoldReplace  lang={}  rms={utterance_rms:.4}  text=\"{}\"",
                                    out.lang, out.text
                                );
                                let _ = result_tx.blocking_send(AsrResult::GoldReplace {
                                    text: out.text,
                                    lang: out.lang,
                                    rms: utterance_rms,
                                });
                            }
                        } else {
                            debug!("GOLD skipped: speech too short ({speech_s:.2}s), likely noise");
                        }
                        gold_buffer.clear();
                        silence_regions.clear();
                        open_silence = None;
                        utterance_start = 0;
                        in_speech = false;
                        silence_samples = 0;
                        partial_fired = false;
                        speech_samples_in_utterance = 0;
                        speech_energy_sum = 0.0;
                        debug!("gold boundary reset");
                    }
                } else {
                    // Pure silence before any speech: discard.
                    trace!("pre-speech silence discarded  rms={energy:.4}");
                }
            }

            Some(AudioMsg::Stop) | None => {
                let gold_s = gold_buffer.len() as f32 / SAMPLE_RATE as f32;
                info!("Stop received  gold_buf={gold_s:.2}s  in_speech={in_speech}");
                if let Some(start) = open_silence.take() {
                    silence_regions.push((start, gold_buffer.len()));
                }
                if !gold_buffer.is_empty() {
                    let speech_s = speech_samples_in_utterance as f32 / SAMPLE_RATE as f32;
                    if speech_samples_in_utterance >= cfg.min_speech_samples {
                        let utterance_rms = (speech_energy_sum
                            / speech_samples_in_utterance as f64)
                            .sqrt() as f32;
                        info!("flushing remaining {gold_s:.2}s as final gold");
                        if let Some(out) = transcribe(&recognizer, &gold_buffer, cfg.min_transcribe_samples) {
                            info!(
                                "→ sending GoldReplace (flush)  lang={}  rms={utterance_rms:.4}  text=\"{}\"",
                                out.lang, out.text
                            );
                            let _ = result_tx.blocking_send(AsrResult::GoldReplace {
                                text: out.text,
                                lang: out.lang,
                                rms: utterance_rms,
                            });
                        }
                    } else {
                        debug!("flush skipped: speech too short ({speech_s:.2}s)");
                    }
                }
                break;
            }
        }
    }
    info!("ASR thread exiting");
}
