//! Domain types to and from the generated proto types.
//!
//! Kept in one file so a schema change surfaces as compile errors in a single
//! place rather than scattered through the daemon and the frontend.

use crate::error::Error;
use crate::pb;
use crate::types::*;

// ---- enums ----------------------------------------------------------------

impl From<Operation> for pb::Operation {
    fn from(o: Operation) -> Self {
        match o {
            Operation::Generate => pb::Operation::Generate,
            Operation::Embed => pb::Operation::Embed,
            Operation::Transcribe => pb::Operation::Transcribe,
        }
    }
}

impl TryFrom<pb::Operation> for Operation {
    type Error = Error;
    fn try_from(o: pb::Operation) -> Result<Self, Error> {
        match o {
            pb::Operation::Generate => Ok(Operation::Generate),
            pb::Operation::Embed => Ok(Operation::Embed),
            pb::Operation::Transcribe => Ok(Operation::Transcribe),
            pb::Operation::Unspecified => Err(Error::InvalidInput("operation not set".into())),
        }
    }
}

/// Proto enums cross the wire as `i32`, and an unknown value is not an error at
/// the codec layer, so every decode goes through here.
pub fn operation_from_i32(v: i32) -> Result<Operation, Error> {
    pb::Operation::try_from(v)
        .map_err(|_| Error::InvalidInput(format!("unknown operation {v}")))
        .and_then(Operation::try_from)
}

impl From<Role> for pb::Role {
    fn from(r: Role) -> Self {
        match r {
            Role::System => pb::Role::System,
            Role::User => pb::Role::User,
            Role::Assistant => pb::Role::Assistant,
        }
    }
}

fn role_from_i32(v: i32) -> Result<Role, Error> {
    match pb::Role::try_from(v) {
        Ok(pb::Role::System) => Ok(Role::System),
        Ok(pb::Role::User) => Ok(Role::User),
        Ok(pb::Role::Assistant) => Ok(Role::Assistant),
        Ok(pb::Role::Unspecified) | Err(_) => Err(Error::InvalidInput(format!("unknown role {v}"))),
    }
}

impl From<FinishReason> for pb::FinishReason {
    fn from(f: FinishReason) -> Self {
        match f {
            FinishReason::Stop => pb::FinishReason::Stop,
            FinishReason::Length => pb::FinishReason::Length,
        }
    }
}

fn finish_from_i32(v: i32) -> Result<FinishReason, Error> {
    match pb::FinishReason::try_from(v) {
        Ok(pb::FinishReason::Stop) => Ok(FinishReason::Stop),
        Ok(pb::FinishReason::Length) => Ok(FinishReason::Length),
        Ok(pb::FinishReason::Unspecified) | Err(_) => {
            Err(Error::InvalidInput(format!("unknown finish reason {v}")))
        }
    }
}

// ---- small structs --------------------------------------------------------

impl From<Usage> for pb::Usage {
    fn from(u: Usage) -> Self {
        pb::Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            reasoning_tokens: u.reasoning_tokens,
        }
    }
}

impl From<pb::Usage> for Usage {
    fn from(u: pb::Usage) -> Self {
        Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            reasoning_tokens: u.reasoning_tokens,
        }
    }
}

impl From<Segment> for pb::Segment {
    fn from(s: Segment) -> Self {
        pb::Segment {
            text: s.text,
            start_s: s.start_s,
            end_s: s.end_s,
        }
    }
}

impl From<pb::Segment> for Segment {
    fn from(s: pb::Segment) -> Self {
        Segment {
            text: s.text,
            start_s: s.start_s,
            end_s: s.end_s,
        }
    }
}

impl From<Image> for pb::Image {
    fn from(i: Image) -> Self {
        pb::Image {
            width: i.width,
            height: i.height,
            rgb8: i.rgb8,
        }
    }
}

impl From<pb::Image> for Image {
    fn from(i: pb::Image) -> Self {
        Image {
            width: i.width,
            height: i.height,
            rgb8: i.rgb8,
        }
    }
}

// ---- messages -------------------------------------------------------------

impl From<Part> for pb::Part {
    fn from(p: Part) -> Self {
        pb::Part {
            part: Some(match p {
                Part::Text(t) => pb::part::Part::Text(t),
                Part::Image(i) => pb::part::Part::Image(i.into()),
            }),
        }
    }
}

impl TryFrom<pb::Part> for Part {
    type Error = Error;
    fn try_from(p: pb::Part) -> Result<Self, Error> {
        match p.part {
            Some(pb::part::Part::Text(t)) => Ok(Part::Text(t)),
            Some(pb::part::Part::Image(i)) => Ok(Part::Image(i.into())),
            None => Err(Error::InvalidInput("message part is empty".into())),
        }
    }
}

impl From<Message> for pb::Message {
    fn from(m: Message) -> Self {
        pb::Message {
            role: pb::Role::from(m.role) as i32,
            parts: m.parts.into_iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<pb::Message> for Message {
    type Error = Error;
    fn try_from(m: pb::Message) -> Result<Self, Error> {
        Ok(Message {
            role: role_from_i32(m.role)?,
            parts: m
                .parts
                .into_iter()
                .map(Part::try_from)
                .collect::<Result<_, _>>()?,
        })
    }
}

impl From<GenerateInput> for pb::GenerateInput {
    fn from(g: GenerateInput) -> Self {
        pb::GenerateInput {
            messages: g.messages.into_iter().map(Into::into).collect(),
            temperature: g.temperature,
            top_p: g.top_p,
            max_tokens: g.max_tokens,
            reasoning: g.reasoning,
        }
    }
}

impl TryFrom<pb::GenerateInput> for GenerateInput {
    type Error = Error;
    fn try_from(g: pb::GenerateInput) -> Result<Self, Error> {
        Ok(GenerateInput {
            messages: g
                .messages
                .into_iter()
                .map(Message::try_from)
                .collect::<Result<_, _>>()?,
            temperature: g.temperature,
            top_p: g.top_p,
            max_tokens: g.max_tokens,
            reasoning: g.reasoning,
        })
    }
}

// ---- inputs ---------------------------------------------------------------

impl From<Input> for pb::Input {
    /// Audio does not travel here. A `Transcribe` input contributes its
    /// `language` to the first message, and its PCM follows as further messages
    /// on the same call.
    fn from(i: Input) -> Self {
        pb::Input {
            input: Some(match i {
                Input::Generate(g) => pb::input::Input::Generate(g.into()),
                Input::Embed { text } => pb::input::Input::Embed(pb::EmbedInput { text }),
                Input::Transcribe(t) => pb::input::Input::Transcribe(pb::TranscribeInput {
                    pcm_s16le: Vec::new(),
                    language: t.language,
                }),
            }),
        }
    }
}

/// A decoded input, with audio left out. The daemon rebuilds the stream from
/// the messages that follow.
#[derive(Debug)]
pub enum DecodedInput {
    Generate(GenerateInput),
    Embed {
        text: String,
    },
    Transcribe {
        language: Option<String>,
        first_chunk: Vec<u8>,
    },
}

impl DecodedInput {
    pub fn operation(&self) -> Operation {
        match self {
            DecodedInput::Generate(_) => Operation::Generate,
            DecodedInput::Embed { .. } => Operation::Embed,
            DecodedInput::Transcribe { .. } => Operation::Transcribe,
        }
    }
}

impl TryFrom<pb::Input> for DecodedInput {
    type Error = Error;
    fn try_from(i: pb::Input) -> Result<Self, Error> {
        match i.input {
            Some(pb::input::Input::Generate(g)) => Ok(DecodedInput::Generate(g.try_into()?)),
            Some(pb::input::Input::Embed(e)) => Ok(DecodedInput::Embed { text: e.text }),
            Some(pb::input::Input::Transcribe(t)) => Ok(DecodedInput::Transcribe {
                language: t.language,
                first_chunk: t.pcm_s16le,
            }),
            None => Err(Error::InvalidInput("input is empty".into())),
        }
    }
}

// ---- outputs and events ---------------------------------------------------

impl From<Output> for pb::Output {
    fn from(o: Output) -> Self {
        pb::Output {
            output: Some(match o {
                Output::Generated {
                    text,
                    reasoning,
                    finish,
                    usage,
                } => pb::output::Output::Generated(pb::Generated {
                    text,
                    reasoning,
                    finish: pb::FinishReason::from(finish) as i32,
                    usage: Some(usage.into()),
                }),
                Output::Embedding { vector, tokens } => {
                    pb::output::Output::Embedding(pb::Embedding { vector, tokens })
                }
                Output::Transcript {
                    text,
                    segments,
                    audio_s,
                } => pb::output::Output::Transcript(pb::Transcript {
                    text,
                    segments: segments.into_iter().map(Into::into).collect(),
                    audio_s,
                }),
            }),
        }
    }
}

impl TryFrom<pb::Output> for Output {
    type Error = Error;
    fn try_from(o: pb::Output) -> Result<Self, Error> {
        match o.output {
            Some(pb::output::Output::Generated(g)) => Ok(Output::Generated {
                text: g.text,
                reasoning: g.reasoning,
                finish: finish_from_i32(g.finish)?,
                usage: g.usage.unwrap_or_default().into(),
            }),
            Some(pb::output::Output::Embedding(e)) => Ok(Output::Embedding {
                vector: e.vector,
                tokens: e.tokens,
            }),
            Some(pb::output::Output::Transcript(t)) => Ok(Output::Transcript {
                text: t.text,
                segments: t.segments.into_iter().map(Into::into).collect(),
                audio_s: t.audio_s,
            }),
            None => Err(Error::InvalidInput("output is empty".into())),
        }
    }
}

impl From<Event> for pb::Event {
    fn from(e: Event) -> Self {
        pb::Event {
            event: Some(match e {
                Event::ReasoningDelta(s) => pb::event::Event::ReasoningDelta(s),
                Event::TextDelta(s) => pb::event::Event::TextDelta(s),
                Event::Segment(s) => pb::event::Event::Segment(s.into()),
                Event::Done { finish, usage } => pb::event::Event::Done(pb::Done {
                    finish: pb::FinishReason::from(finish) as i32,
                    usage: Some(usage.into()),
                }),
            }),
        }
    }
}

impl TryFrom<pb::Event> for Event {
    type Error = Error;
    fn try_from(e: pb::Event) -> Result<Self, Error> {
        match e.event {
            Some(pb::event::Event::ReasoningDelta(s)) => Ok(Event::ReasoningDelta(s)),
            Some(pb::event::Event::TextDelta(s)) => Ok(Event::TextDelta(s)),
            Some(pb::event::Event::Segment(s)) => Ok(Event::Segment(s.into())),
            Some(pb::event::Event::Done(d)) => Ok(Event::Done {
                finish: finish_from_i32(d.finish)?,
                usage: d.usage.unwrap_or_default().into(),
            }),
            None => Err(Error::InvalidInput("event is empty".into())),
        }
    }
}

// ---- model info -----------------------------------------------------------

impl From<ModelInfo> for pb::ModelInfo {
    fn from(m: ModelInfo) -> Self {
        let (state, failure) = match &m.state {
            ModelState::Loading => (pb::ModelState::Loading, String::new()),
            ModelState::Ready => (pb::ModelState::Ready, String::new()),
            ModelState::Failed(why) => (pb::ModelState::Failed, why.clone()),
            ModelState::Unavailable => (pb::ModelState::Unavailable, String::new()),
        };
        pb::ModelInfo {
            id: m.id,
            operations: m
                .operations
                .into_iter()
                .map(|o| pb::Operation::from(o) as i32)
                .collect(),
            state: state as i32,
            failure,
            loaded_at: m.loaded_at,
            image_input: m
                .image_input
                .map(|(width, height)| pb::ImageInputSize { width, height }),
            reasoning: m.reasoning,
        }
    }
}

impl TryFrom<pb::ModelInfo> for ModelInfo {
    type Error = Error;
    fn try_from(m: pb::ModelInfo) -> Result<Self, Error> {
        let state = match pb::ModelState::try_from(m.state) {
            Ok(pb::ModelState::Loading) => ModelState::Loading,
            Ok(pb::ModelState::Ready) => ModelState::Ready,
            Ok(pb::ModelState::Failed) => ModelState::Failed(m.failure),
            Ok(pb::ModelState::Unavailable) => ModelState::Unavailable,
            Ok(pb::ModelState::Unspecified) | Err(_) => {
                return Err(Error::InvalidInput(format!(
                    "unknown model state {}",
                    m.state
                )))
            }
        };
        Ok(ModelInfo {
            id: m.id,
            operations: m
                .operations
                .into_iter()
                .map(operation_from_i32)
                .collect::<Result<_, _>>()?,
            state,
            loaded_at: m.loaded_at,
            image_input: m.image_input.map(|s| (s.width, s.height)),
            reasoning: m.reasoning,
        })
    }
}
