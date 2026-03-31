use serde::{Deserialize, Serialize};

/// Messages sent from the client to the server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Sent once before streaming begins.
    Start,
    /// Sent when the user stops recording.
    Stop,
}

/// Messages sent from the server to the client.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Recognizer is loaded and ready to accept audio.
    Ready,
    /// Quick transcription of the most recent utterance.
    /// May be superseded by a subsequent GoldReplace.
    Partial {
        id: String,
        text: String,
        /// Detected language, e.g. "zh", "en", "ja".
        lang: String,
        /// RMS energy of the speech portion of this utterance.
        rms: f32,
    },
    /// Accurate transcription covering everything since the last gold boundary.
    /// Clients should replace all pending partials with this text.
    GoldReplace {
        text: String,
        /// Detected language of the last utterance in this gold window.
        lang: String,
        /// RMS energy of the last utterance's speech portion.
        rms: f32,
    },
}
