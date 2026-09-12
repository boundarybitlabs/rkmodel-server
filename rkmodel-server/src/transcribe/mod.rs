//! Transcription, proxied to rkwhisperd.
//!
//! There is no worker or queue here. rkwhisperd owns the whisper models and
//! does its own queueing, so this module only translates: audio in as this
//! protocol's chunks, out through `rkwhisper-client`, and rkwhisperd's
//! responses back as [`Event`]s.
//!
//! [`Asr`] is to this module what `worker::backend::Backend` is to generate.
//! Above the trait is ordinary Rust with tests; below it is a Unix socket and a
//! shared-memory ring that only exist where rkwhisperd runs.

pub mod probe;
pub mod rkwhisper;

use std::sync::Arc;

use rkmodel_server_protocol::{transcript, ByteStream, Error, Event, FinishReason, Segment, Usage};
use rkwhisper_client::Response;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;

/// Events buffered between the session and the gRPC stream. Segments arrive as
/// whole utterances rather than token by token, so this stays small.
const EVENT_BUFFER: usize = 16;

/// One open session with rkwhisperd, split so audio and responses move
/// independently. The client's own docs call this the shape live transcription
/// wants: `send_audio` waits for room in the ring, and nothing should hold up
/// the segments coming back while it does.
pub type Halves = (Box<dyn AsrSender>, Box<dyn AsrReceiver>);

/// Opens sessions against whatever is serving whisper models.
#[async_trait::async_trait]
pub trait Asr: Send + Sync {
    /// Opens one session. `language` unset takes rkwhisper's default, `en`.
    async fn open(&self, model: &str, language: Option<&str>) -> Result<Halves, Error>;

    /// Whether this model can be transcribed on right now.
    ///
    /// Opening a session and dropping it is exactly that question: the
    /// handshake names the model, and no audio follows, so rkwhisperd does no
    /// work for it. See [`probe`].
    async fn probe(&self, model: &str) -> Result<(), Error> {
        self.open(model, None).await.map(|_| ())
    }
}

#[async_trait::async_trait]
pub trait AsrSender: Send {
    async fn send_audio(&mut self, pcm: &[u8]) -> Result<(), Error>;
    /// Tells rkwhisperd no more audio is coming.
    async fn finish(&mut self) -> Result<(), Error>;
}

#[async_trait::async_trait]
pub trait AsrReceiver: Send {
    /// rkwhisperd's next response, as it sent it.
    async fn recv(&mut self) -> Result<Response, Error>;
}

/// Aborts both halves of a session when it drops.
///
/// Dropping the pump drops the sender, which closes the socket's write half,
/// and rkwhisperd cancels the job. That is what makes a client disconnect stop
/// the work: tonic drops the response stream, which drops this.
#[derive(Debug)]
pub struct SessionHandle {
    pump: JoinHandle<()>,
    reader: JoinHandle<()>,
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.pump.abort();
        self.reader.abort();
    }
}

/// Opens a session and starts both halves running.
///
/// Opening is awaited rather than deferred into the stream, so an unreachable
/// rkwhisperd or a model it does not serve fails the call itself, the way a
/// full queue does for generate.
pub async fn start(
    asr: Arc<dyn Asr>,
    model: &str,
    language: Option<String>,
    mut audio: ByteStream,
) -> Result<(mpsc::Receiver<Result<Event, Error>>, SessionHandle), Error> {
    let (mut sender, mut receiver) = asr.open(model, language.as_deref()).await?;
    let (events, rx) = mpsc::channel(EVENT_BUFFER);

    let pump = tokio::spawn(async move {
        while let Some(chunk) = audio.next().await {
            match chunk {
                Ok(pcm) => {
                    if let Err(e) = sender.send_audio(&pcm).await {
                        tracing::warn!("sending audio to rkwhisperd failed: {e}");
                        return;
                    }
                }
                // The frontend failed to decode the upload. Returning drops the
                // sender, which cancels the job rather than letting rkwhisperd
                // transcribe a truncated clip and call it complete.
                Err(e) => {
                    tracing::warn!("the audio stream failed: {e}");
                    return;
                }
            }
        }
        if let Err(e) = sender.finish().await {
            tracing::warn!("closing the audio stream failed: {e}");
        }
    });

    let model_for_log = model.to_string();
    let reader = tokio::spawn(async move {
        loop {
            // Cancellation reaches this loop two ways: the client turns
            // rkwhisperd's `Cancelled` into an error of its own, but a response
            // that arrives unconverted maps to the same marker. Folding both
            // into one `Result` keeps them from being handled differently.
            let mapped = match receiver.recv().await {
                Ok(response) => map_response(response),
                Err(e) => Err(e),
            };

            let event = match mapped {
                Ok(Some(e)) => e,
                Ok(None) => continue,
                // We asked for this by dropping the sender, so it ends the
                // stream rather than failing it.
                Err(e) if is_cancelled(&e) => return,
                Err(e) => {
                    let _ = events.send(Err(e)).await;
                    return;
                }
            };

            let done = matches!(event, Event::Done { .. });
            if events.send(Ok(event)).await.is_err() {
                tracing::debug!(model = %model_for_log, "transcription dropped by its caller");
                return;
            }
            if done {
                return;
            }
        }
    });

    Ok((rx, SessionHandle { pump, reader }))
}

/// The marker a cancellation carries, which is an acknowledgement of something
/// this daemon asked for rather than a failure to report.
pub(crate) const CANCELLED: &str = "rkwhisper session cancelled";

fn is_cancelled(e: &Error) -> bool {
    matches!(e, Error::Runtime { call, .. } if call == CANCELLED)
}

/// rkwhisperd's responses, as this protocol's events.
///
/// `Ok(None)` is a response with nothing for the caller: the VAD boundaries
/// have no counterpart here, and `ServerHello` belongs to the handshake the
/// client already consumed.
fn map_response(response: Response) -> Result<Option<Event>, Error> {
    match response {
        Response::Segment { text, begin, end } => Ok(Some(Event::Segment(Segment {
            text,
            start_s: begin,
            end_s: end,
        }))),
        // Whisper reports no token counts, so usage stays zero rather than
        // carrying a number this daemon made up. rkwhisperd's own timing is
        // logged instead: it is the first thing to look at when a transcription
        // feels slow, and an `rtf` over 1 means slower than real time.
        Response::Done { audio_s, rtf } => {
            tracing::debug!(audio_s, rtf, "rkwhisperd finished a transcription");
            Ok(Some(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }))
        }
        Response::BackOff { retry_after_ms, .. } => Err(Error::Busy { retry_after_ms }),
        Response::Error { error } => Err(Error::Runtime {
            call: format!("rkwhisperd: {error}"),
            code: -1,
        }),
        Response::Cancelled { .. } => Err(Error::Runtime {
            call: CANCELLED.into(),
            code: -1,
        }),
        Response::SpeechStarted { .. } | Response::SpeechEnded { .. } => Ok(None),
        Response::ServerHello(_) => Ok(None),
    }
}

/// Folds a session's events into the one output the unary call returns.
///
/// `audio_s` comes from the frontend, which decoded the upload and so knows the
/// duration exactly. rkwhisperd's own figure covers what its VAD kept.
pub async fn collect(
    mut rx: mpsc::Receiver<Result<Event, Error>>,
    handle: SessionHandle,
    audio_s: f32,
) -> Result<rkmodel_server_protocol::Output, Error> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut done = false;

    while let Some(event) = rx.recv().await {
        match event? {
            Event::Segment(s) => segments.push(s),
            Event::Done { .. } => done = true,
            Event::TextDelta(_) | Event::ReasoningDelta(_) | Event::ToolCall(_) => {}
        }
    }
    drop(handle);

    if !done {
        return Err(Error::Runtime {
            call: "transcription ended without a final event".into(),
            code: -1,
        });
    }

    Ok(rkmodel_server_protocol::Output::Transcript {
        text: transcript(&segments),
        segments,
        audio_s,
    })
}

#[cfg(test)]
mod tests;
