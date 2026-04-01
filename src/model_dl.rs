use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

use crate::asr::ModelPaths;

const DEFAULT_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2";
const DEFAULT_MODEL_PATH: &str =
    "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx";
const DEFAULT_TOKENS_PATH: &str =
    "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt";
const ARCHIVE_NAME: &str =
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2";

/// Returns ready-to-use model paths, downloading the default bundle if needed.
///
/// `model_path` / `tokens_path` come from the resolved config (already merged
/// with env vars and JSON file). If both point to existing files, no download
/// occurs. Otherwise the default SenseVoice bundle is fetched into `models/`.
pub fn ensure_model_paths(
    num_threads: i32,
    model_path: Option<String>,
    tokens_path: Option<String>,
) -> Result<ModelPaths> {
    let model = model_path.unwrap_or_else(|| DEFAULT_MODEL_PATH.to_string());
    let tokens = tokens_path.unwrap_or_else(|| DEFAULT_TOKENS_PATH.to_string());

    if Path::new(&model).exists() && Path::new(&tokens).exists() {
        return Ok(ModelPaths { model, tokens, num_threads });
    }

    info!("Model files not found, downloading default paraformer-zh bundle (~214 MB)...");
    std::fs::create_dir_all("models").context("Failed to create models/ directory")?;

    let archive_path = format!("models/{ARCHIVE_NAME}");
    // Remove a stale partial archive from a previous failed attempt, if any.
    let _ = std::fs::remove_file(&archive_path);

    download_with_progress(DEFAULT_URL, &archive_path)?;
    extract_tar_bz2(&archive_path, "models")?;
    let _ = std::fs::remove_file(&archive_path);

    info!("Model ready.");
    Ok(ModelPaths {
        model: DEFAULT_MODEL_PATH.to_string(),
        tokens: DEFAULT_TOKENS_PATH.to_string(),
        num_threads,
    })
}

fn download_with_progress(url: &str, dest: &str) -> Result<()> {
    let resp = ureq::get(url).call().context("HTTP request failed")?;
    let total_mb = resp
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok())
        .map(|b| b / (1024 * 1024));

    let mut reader = resp.into_reader();
    let mut file =
        std::fs::File::create(dest).with_context(|| format!("Cannot create {dest}"))?;

    let mut buf = [0u8; 64 * 1024];
    let mut downloaded_bytes: u64 = 0;
    let mut last_reported_mb: u64 = 0;

    loop {
        let n = reader.read(&mut buf).context("Download read error")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("Disk write error")?;
        downloaded_bytes += n as u64;

        let mb = downloaded_bytes / (1024 * 1024);
        if mb > last_reported_mb {
            match total_mb {
                Some(t) => info!("  Downloading... {mb} / {t} MB"),
                None => info!("  Downloading... {mb} MB"),
            }
            last_reported_mb = mb;
        }
    }

    info!("Download complete ({} MB).", downloaded_bytes / (1024 * 1024));
    Ok(())
}

fn extract_tar_bz2(archive: &str, dest: &str) -> Result<()> {
    info!("Extracting archive...");
    let file =
        std::fs::File::open(archive).with_context(|| format!("Cannot open {archive}"))?;
    let bz2 = bzip2::read::BzDecoder::new(file);
    let mut tar = tar::Archive::new(bz2);
    tar.unpack(dest).context("Extraction failed")?;
    info!("Extraction complete.");
    Ok(())
}
