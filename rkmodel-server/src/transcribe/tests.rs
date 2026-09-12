use std::sync::{Arc, Mutex};

use super::*;
use rkwhisper_client::Response;

/// The model and language each `open` was called with.
type Opened = Arc<Mutex<Vec<(String, Option<String>)>>>;

/// Replays scripted responses, and records the audio it was handed.
struct FakeAsr {
    script: Vec<Result<Response, Error>>,
    sent: Arc<Mutex<Vec<u8>>>,
    finished: Arc<Mutex<bool>>,
    open_error: Option<Error>,
    seen: Opened,
}

impl FakeAsr {
    fn new(script: Vec<Response>) -> FakeAsr {
        FakeAsr {
            script: script.into_iter().map(Ok).collect(),
            sent: Arc::new(Mutex::new(Vec::new())),
            finished: Arc::new(Mutex::new(false)),
            open_error: None,
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn refusing(error: Error) -> FakeAsr {
        FakeAsr {
            open_error: Some(error),
            ..FakeAsr::new(vec![])
        }
    }
}

#[async_trait::async_trait]
impl Asr for FakeAsr {
    async fn open(&self, model: &str, language: Option<&str>) -> Result<Halves, Error> {
        self.seen
            .lock()
            .unwrap()
            .push((model.to_string(), language.map(str::to_string)));
        if let Some(e) = &self.open_error {
            return Err(e.clone());
        }
        Ok((
            Box::new(FakeSender {
                sent: self.sent.clone(),
                finished: self.finished.clone(),
            }),
            Box::new(FakeReceiver {
                script: self.script.clone(),
            }),
        ))
    }
}

struct FakeSender {
    sent: Arc<Mutex<Vec<u8>>>,
    finished: Arc<Mutex<bool>>,
}

#[async_trait::async_trait]
impl AsrSender for FakeSender {
    async fn send_audio(&mut self, pcm: &[u8]) -> Result<(), Error> {
        self.sent.lock().unwrap().extend_from_slice(pcm);
        Ok(())
    }
    async fn finish(&mut self) -> Result<(), Error> {
        *self.finished.lock().unwrap() = true;
        Ok(())
    }
}

struct FakeReceiver {
    script: Vec<Result<Response, Error>>,
}

#[async_trait::async_trait]
impl AsrReceiver for FakeReceiver {
    async fn recv(&mut self) -> Result<Response, Error> {
        if self.script.is_empty() {
            // A real receiver blocks here. Nothing in these tests reads past
            // the scripted end, and the handle aborts the task either way.
            std::future::pending::<()>().await;
        }
        self.script.remove(0)
    }
}

fn segment(text: &str, begin: f32, end: f32) -> Response {
    Response::Segment {
        text: text.to_string(),
        begin,
        end,
    }
}

fn done() -> Response {
    Response::Done {
        audio_s: 3.0,
        rtf: 0.2,
    }
}

fn audio(chunks: &[&[u8]]) -> ByteStream {
    let chunks: Vec<Result<Vec<u8>, Error>> = chunks.iter().map(|c| Ok(c.to_vec())).collect();
    Box::pin(tokio_stream::iter(chunks))
}

async fn drain(mut rx: mpsc::Receiver<Result<Event, Error>>) -> Vec<Result<Event, Error>> {
    let mut out = Vec::new();
    while let Some(event) = rx.recv().await {
        let done = matches!(event, Ok(Event::Done { .. })) || event.is_err();
        out.push(event);
        if done {
            break;
        }
    }
    out
}

// ---- the response mapping, which is the table in MODEL_SERVER.md ----------

#[test]
fn a_segment_becomes_a_segment_event() {
    let event = map_response(segment(" the sky is blue", 0.5, 1.25)).unwrap();
    assert_eq!(
        event,
        Some(Event::Segment(Segment {
            text: " the sky is blue".into(),
            start_s: 0.5,
            end_s: 1.25,
        }))
    );
}

#[test]
fn done_becomes_done_with_no_token_counts() {
    let event = map_response(done()).unwrap().unwrap();
    match event {
        Event::Done { finish, usage } => {
            assert_eq!(finish, FinishReason::Stop);
            assert_eq!(usage, Usage::default());
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[test]
fn backoff_becomes_busy_carrying_its_delay() {
    let err = map_response(Response::BackOff {
        reason: "all workers busy".into(),
        retry_after_ms: 250,
    })
    .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Busy {
                retry_after_ms: 250
            }
        ),
        "{err:?}"
    );
}

#[test]
fn an_error_response_becomes_runtime_naming_rkwhisperd() {
    let err = map_response(Response::Error {
        error: "model not loaded".into(),
    })
    .unwrap_err();
    match err {
        Error::Runtime { call, .. } => assert!(call.contains("model not loaded"), "{call}"),
        other => panic!("expected Runtime, got {other:?}"),
    }
}

#[test]
fn the_vad_boundaries_carry_no_event() {
    assert_eq!(
        map_response(Response::SpeechStarted { begin: 0.1 }).unwrap(),
        None
    );
    assert_eq!(
        map_response(Response::SpeechEnded { end: 2.0 }).unwrap(),
        None
    );
}

// ---- joining segments ------------------------------------------------------

#[test]
fn segments_join_into_one_transcript_without_doubling_spaces() {
    let segments = vec![
        Segment {
            text: " the sky".into(),
            start_s: 0.0,
            end_s: 1.0,
        },
        Segment {
            text: " is blue".into(),
            start_s: 1.0,
            end_s: 2.0,
        },
    ];
    assert_eq!(joined(&segments), "the sky is blue");
}

#[test]
fn an_empty_segment_does_not_become_a_stray_space() {
    let segments = vec![
        Segment {
            text: "hello".into(),
            start_s: 0.0,
            end_s: 1.0,
        },
        Segment {
            text: "   ".into(),
            start_s: 1.0,
            end_s: 2.0,
        },
    ];
    assert_eq!(joined(&segments), "hello");
}

// ---- driving a session -----------------------------------------------------

#[tokio::test]
async fn a_session_forwards_every_chunk_and_ends_with_done() {
    let fake = Arc::new(FakeAsr::new(vec![
        segment(" the sky", 0.0, 1.0),
        segment(" is blue", 1.0, 2.0),
        done(),
    ]));
    let sent = fake.sent.clone();
    let finished = fake.finished.clone();

    let (rx, handle) = start(fake, "whisper-small-30s", None, audio(&[b"ab", b"cd"]))
        .await
        .unwrap();
    let events = drain(rx).await;
    drop(handle);

    assert_eq!(events.len(), 3);
    assert!(
        matches!(events[2], Ok(Event::Done { .. })),
        "{:?}",
        events[2]
    );
    assert_eq!(&*sent.lock().unwrap(), b"abcd");
    assert!(*finished.lock().unwrap(), "the sender should have finished");
}

#[tokio::test]
async fn the_model_and_language_reach_the_session() {
    let fake = Arc::new(FakeAsr::new(vec![done()]));
    let seen = fake.seen.clone();

    let (rx, handle) = start(
        fake,
        "whisper-small-30s",
        Some("fr".to_string()),
        audio(&[b"ab"]),
    )
    .await
    .unwrap();
    drain(rx).await;
    drop(handle);

    assert_eq!(
        &*seen.lock().unwrap(),
        &[("whisper-small-30s".to_string(), Some("fr".to_string()))]
    );
}

#[tokio::test]
async fn a_refused_session_fails_the_call_rather_than_the_stream() {
    let fake = Arc::new(FakeAsr::refusing(Error::Unreachable("no socket".into())));
    let err = start(fake, "whisper-small-30s", None, audio(&[b"ab"]))
        .await
        .expect_err("opening should fail");
    assert!(matches!(err, Error::Unreachable(_)), "{err:?}");
}

#[tokio::test]
async fn a_backoff_mid_session_reaches_the_caller_as_busy() {
    let fake = Arc::new(FakeAsr::new(vec![Response::BackOff {
        reason: "all workers busy".into(),
        retry_after_ms: 500,
    }]));
    let (rx, handle) = start(fake, "whisper-small-30s", None, audio(&[b"ab"]))
        .await
        .unwrap();
    let events = drain(rx).await;
    drop(handle);

    assert_eq!(events.len(), 1);
    match &events[0] {
        Err(Error::Busy { retry_after_ms }) => assert_eq!(*retry_after_ms, 500),
        other => panic!("expected Busy, got {other:?}"),
    }
}

#[tokio::test]
async fn a_cancelled_session_ends_the_stream_without_an_error() {
    let fake = Arc::new(FakeAsr::new(vec![Response::Cancelled {
        audio_s: 1.0,
        rtf: 0.1,
        windows_dispatched: 2,
        windows_completed: 1,
    }]));
    let (rx, handle) = start(fake, "whisper-small-30s", None, audio(&[b"ab"]))
        .await
        .unwrap();
    let events = drain(rx).await;
    drop(handle);

    assert!(events.is_empty(), "{events:?}");
}

#[tokio::test]
async fn collect_folds_segments_into_one_transcript() {
    let fake = Arc::new(FakeAsr::new(vec![
        segment(" the sky", 0.0, 1.0),
        segment(" is blue", 1.0, 2.0),
        done(),
    ]));
    let (rx, handle) = start(fake, "whisper-small-30s", None, audio(&[b"ab"]))
        .await
        .unwrap();
    let output = collect(rx, handle, 2.5).await.unwrap();

    match output {
        rkmodel_server_protocol::Output::Transcript {
            text,
            segments,
            audio_s,
        } => {
            assert_eq!(text, "the sky is blue");
            assert_eq!(segments.len(), 2);
            assert_eq!(audio_s, 2.5);
        }
        other => panic!("expected Transcript, got {other:?}"),
    }
}

#[tokio::test]
async fn collect_refuses_a_session_that_ended_without_done() {
    // rkwhisperd going away mid-transcription: segments arrived, the final
    // event never did. A partial transcript must not be returned as a whole one.
    let (tx, rx) = mpsc::channel(4);
    tx.send(Ok(Event::Segment(Segment {
        text: " the sky".into(),
        start_s: 0.0,
        end_s: 1.0,
    })))
    .await
    .unwrap();
    drop(tx);

    let err = collect(rx, idle_handle(), 2.5)
        .await
        .expect_err("a session with no final event should fail");
    match err {
        Error::Runtime { call, .. } => assert!(call.contains("without a final event"), "{call}"),
        other => panic!("expected Runtime, got {other:?}"),
    }
}

/// A handle over two tasks that do nothing, for the cases that drive `collect`
/// through a channel rather than a session.
fn idle_handle() -> SessionHandle {
    SessionHandle {
        pump: tokio::spawn(std::future::pending()),
        reader: tokio::spawn(std::future::pending()),
    }
}
