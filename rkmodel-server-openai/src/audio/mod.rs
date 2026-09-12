//! `POST /v1/audio/transcriptions`.
//!
//! The upload arrives as multipart, is decoded to 16 kHz mono s16le here, and
//! goes to the daemon as a stream rather than a buffer. Everything below the
//! parsing is pure, so the five response formats are tested without a daemon or
//! a socket.

pub mod decode;

use axum_extra::extract::Multipart;
use rkmodel_server_protocol::Segment;
use serde_json::{json, Value};

use crate::error::ApiError;

/// OpenAI's cap on the upload, which this matches so a client that already
/// checks against it sees the same answer here.
pub const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

/// The cap on the whole multipart body: the file, plus boundaries, headers and
/// the other fields.
///
/// Axum's own default is 2 MB, which is far under the file limit above and
/// would refuse most real uploads before any of this code saw them, so the
/// route sets this explicitly. The slack is for the framing, which is a few
/// hundred bytes in practice.
pub const MAX_BODY_BYTES: usize = MAX_UPLOAD_BYTES + 1024 * 1024;

/// What the caller asked to get back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

impl ResponseFormat {
    fn parse(value: &str) -> Result<ResponseFormat, ApiError> {
        match value {
            "json" => Ok(ResponseFormat::Json),
            "text" => Ok(ResponseFormat::Text),
            "verbose_json" => Ok(ResponseFormat::VerboseJson),
            "srt" => Ok(ResponseFormat::Srt),
            "vtt" => Ok(ResponseFormat::Vtt),
            other => Err(ApiError::invalid_request(
                format!(
                    "Unknown response_format {other:?}. Use json, text, verbose_json, srt or vtt."
                ),
                Some("response_format"),
            )),
        }
    }

    /// The `Content-Type` its body goes out with.
    pub fn content_type(self) -> &'static str {
        match self {
            ResponseFormat::Json | ResponseFormat::VerboseJson => "application/json",
            // OpenAI sends all three of these as plain text.
            ResponseFormat::Text | ResponseFormat::Srt | ResponseFormat::Vtt => {
                "text/plain; charset=utf-8"
            }
        }
    }
}

/// One parsed upload.
pub struct TranscriptionRequest {
    pub model: String,
    pub file: Vec<u8>,
    /// Only ever a hint to the decoder's probe.
    pub filename: Option<String>,
    pub language: Option<String>,
    pub format: ResponseFormat,
}

/// Reads the multipart body.
///
/// `prompt` and `temperature` are accepted and ignored: rkwhisper takes no
/// prompt and decodes with beam search, so honouring either would be a lie.
pub async fn parse(mut form: Multipart) -> Result<TranscriptionRequest, ApiError> {
    let mut model = None;
    let mut file = None;
    let mut filename = None;
    let mut language = None;
    let mut format = ResponseFormat::Json;

    while let Some(field) = form.next_field().await.map_err(malformed)? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                filename = field.file_name().map(str::to_string);
                let bytes = field.bytes().await.map_err(malformed)?;
                if bytes.len() > MAX_UPLOAD_BYTES {
                    return Err(ApiError::too_large(format!(
                        "The audio file is {} bytes, over the {MAX_UPLOAD_BYTES} byte limit.",
                        bytes.len()
                    )));
                }
                file = Some(bytes.to_vec());
            }
            "model" => model = Some(text(field).await?),
            "response_format" => format = ResponseFormat::parse(&text(field).await?)?,
            "language" => {
                let value = text(field).await?;
                // An empty field is how some clients spell "unset".
                language = (!value.is_empty()).then_some(value);
            }
            "stream" => {
                if text(field).await?.trim() == "true" {
                    return Err(ApiError::invalid_request(
                        "Streaming transcriptions are not supported yet.",
                        Some("stream"),
                    ));
                }
            }
            // rkwhisper has no prompt input and decodes with beam search, so
            // these change nothing. Ignored rather than refused, because
            // refusing them would break clients that always send them.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let model =
        model.ok_or_else(|| ApiError::invalid_request("model is required.", Some("model")))?;
    let file = file.ok_or_else(|| ApiError::invalid_request("file is required.", Some("file")))?;
    if file.is_empty() {
        return Err(ApiError::invalid_request(
            "The audio file is empty.",
            Some("file"),
        ));
    }

    Ok(TranscriptionRequest {
        model,
        file,
        filename,
        language,
        format,
    })
}

/// A multipart failure, with the status the error itself reports.
///
/// Body-too-large arrives here rather than as a size check of our own, since
/// the limit is enforced while the body is still being read. Reporting it as a
/// malformed request would send a caller looking for a fault in their encoding
/// rather than at the size of their file.
fn malformed(e: axum_extra::extract::multipart::MultipartError) -> ApiError {
    if e.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        return ApiError::too_large(format!(
            "The upload is over the {MAX_UPLOAD_BYTES} byte limit."
        ));
    }
    ApiError::invalid_request(format!("The upload is malformed: {e}"), None)
}

async fn text(field: axum_extra::extract::multipart::Field) -> Result<String, ApiError> {
    field
        .text()
        .await
        .map(|s| s.trim().to_string())
        .map_err(|e| ApiError::invalid_request(format!("A form field is malformed: {e}"), None))
}

/// The transcript, in whichever format was asked for.
pub fn body(
    format: ResponseFormat,
    text: &str,
    segments: &[Segment],
    duration_s: f32,
    language: Option<&str>,
) -> String {
    match format {
        ResponseFormat::Json => json!({ "text": text }).to_string(),
        ResponseFormat::Text => format!("{text}\n"),
        ResponseFormat::VerboseJson => verbose(text, segments, duration_s, language).to_string(),
        ResponseFormat::Srt => srt(segments),
        ResponseFormat::Vtt => vtt(segments),
    }
}

/// `verbose_json`, with every field the official SDK's model declares.
///
/// rkwhisper reports no per-segment confidence, so `avg_logprob`,
/// `compression_ratio` and `no_speech_prob` are zeros standing in for numbers
/// that do not exist rather than measurements. `tokens` is empty for the same
/// reason. They are present because the SDK's schema requires them.
fn verbose(text: &str, segments: &[Segment], duration_s: f32, language: Option<&str>) -> Value {
    let segments: Vec<Value> = segments
        .iter()
        .enumerate()
        .map(|(i, s)| {
            json!({
                "id": i,
                "seek": 0,
                "start": s.start_s,
                "end": s.end_s,
                "text": s.text,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            })
        })
        .collect();

    json!({
        "task": "transcribe",
        // rkwhisper's own default when the request names none.
        "language": language.unwrap_or("en"),
        "duration": duration_s,
        "text": text,
        "segments": segments,
    })
}

fn srt(segments: &[Segment]) -> String {
    let mut out = String::new();
    for (i, segment) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            timestamp(segment.start_s, ','),
            timestamp(segment.end_s, ',')
        ));
        out.push_str(segment.text.trim());
        out.push_str("\n\n");
    }
    out
}

fn vtt(segments: &[Segment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for segment in segments {
        out.push_str(&format!(
            "{} --> {}\n",
            timestamp(segment.start_s, '.'),
            timestamp(segment.end_s, '.')
        ));
        out.push_str(segment.text.trim());
        out.push_str("\n\n");
    }
    out
}

/// `HH:MM:SS,mmm` for SubRip, `HH:MM:SS.mmm` for WebVTT. The two formats differ
/// only in that separator.
fn timestamp(seconds: f32, separator: char) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        total_s / 3600,
        (total_s / 60) % 60,
        total_s % 60,
        ms
    )
}

/// A RIFF/WAVE file over `samples`, which is the one container these tests can
/// build by hand and symphonia reads with its default features. Shared with the
/// route tests, which need an upload that actually decodes.
#[cfg(test)]
pub(crate) fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
    let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let block_align = channels * 2;
    let byte_rate = rate * block_align as u32;

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // format: PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// A sine at `hz`, which survives resampling in a way silence or a ramp does
/// not.
#[cfg(test)]
pub(crate) fn sine(rate: u32, hz: f32, seconds: f32) -> Vec<i16> {
    let count = (rate as f32 * seconds) as usize;
    (0..count)
        .map(|i| {
            let t = i as f32 / rate as f32;
            ((t * hz * std::f32::consts::TAU).sin() * 0.5 * i16::MAX as f32) as i16
        })
        .collect()
}

#[cfg(test)]
mod tests;
