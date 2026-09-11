//! The daemon's vocabulary, free of proto details.

use crate::ByteStream;

/// The fixed set of inference jobs. Adding one is a protocol change; adding a
/// model is a config change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    Generate,
    Embed,
    Transcribe,
}

impl Operation {
    pub fn as_str(self) -> &'static str {
        match self {
            Operation::Generate => "generate",
            Operation::Embed => "embed",
            Operation::Transcribe => "transcribe",
        }
    }
}

impl std::str::FromStr for Operation {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "generate" => Ok(Operation::Generate),
            "embed" => Ok(Operation::Embed),
            "transcribe" => Ok(Operation::Transcribe),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Decoded by the frontend. The daemon resizes and normalizes for its encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgb8: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Text(String),
    Image(Image),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Message {
            role,
            parts: vec![Part::Text(text.into())],
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerateInput {
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    /// `None` takes the model's configured default.
    pub reasoning: Option<bool>,
}

/// Audio arrives as later messages on the same call, not as a field, so this
/// carries the stream rather than a buffer.
pub struct TranscribeInput {
    /// 16 kHz mono signed 16-bit little-endian, in chunks sent as they are
    /// decoded.
    pub pcm_s16le: ByteStream,
    pub language: Option<String>,
}

impl std::fmt::Debug for TranscribeInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscribeInput")
            .field("pcm_s16le", &"<stream>")
            .field("language", &self.language)
            .finish()
    }
}

#[derive(Debug)]
pub enum Input {
    Generate(GenerateInput),
    Embed { text: String },
    Transcribe(TranscribeInput),
}

impl Input {
    /// The operation this input belongs to. The daemon refuses a call whose
    /// input variant does not match the operation named.
    pub fn operation(&self) -> Operation {
        match self {
            Input::Generate(_) => Operation::Generate,
            Input::Embed { .. } => Operation::Embed,
            Input::Transcribe(_) => Operation::Transcribe,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub reasoning_tokens: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub text: String,
    pub start_s: f32,
    pub end_s: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    Generated {
        text: String,
        reasoning: Option<String>,
        finish: FinishReason,
        usage: Usage,
    },
    Embedding {
        vector: Vec<f32>,
        tokens: u32,
    },
    Transcript {
        text: String,
        segments: Vec<Segment>,
        audio_s: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    ReasoningDelta(String),
    TextDelta(String),
    Segment(Segment),
    Done { finish: FinishReason, usage: Usage },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    Loading,
    Ready,
    Failed(String),
    Unavailable,
}

impl ModelState {
    /// What `GET /health` and `GET /v1/models` report.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelState::Loading => "loading",
            ModelState::Ready => "ready",
            ModelState::Failed(_) => "failed",
            ModelState::Unavailable => "unavailable",
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, ModelState::Ready)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub operations: Vec<Operation>,
    pub state: ModelState,
    /// Unix seconds, and the `created` field of `GET /v1/models`.
    pub loaded_at: u64,
    /// The vision encoder's input size, for models that take images. The
    /// frontend downsizes to it, which keeps image payloads small.
    pub image_input: Option<(u32, u32)>,
    pub reasoning: bool,
}

impl ModelInfo {
    pub fn offers(&self, operation: Operation) -> bool {
        self.operations.contains(&operation)
    }
}
