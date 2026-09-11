//! One worker thread and one bounded queue per loaded model.
//!
//! `RkllmSession` already serializes runs with an internal lock, so concurrent
//! calls would queue correctly without any of this. The explicit queue exists
//! for three things that lock cannot do:
//!
//! - **Refuse work.** A full queue answers straight away, rather than parking
//!   another blocking thread on a mutex.
//! - **Skip cancelled requests.** A request whose client left while it waited
//!   is dropped when it reaches the front, before it spends seconds in prefill.
//! - **Know which request is running**, which cancellation needs.

pub mod backend;
pub mod rkllm;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

use rkmodel_server_protocol::{Error, Event, FinishReason, Usage};
use tokio::sync::mpsc;

use crate::config::Sampling;
use crate::generate::reasoning::{Piece, ReasoningParser};

pub use backend::Backend;

/// What a client is told to wait before retrying a full queue. The daemon
/// cannot know how long the run in front will take, so this is a floor that
/// keeps a rejected client from spinning.
const BUSY_RETRY_AFTER_MS: u32 = 1_000;

/// How many events may sit between the worker thread and the connection.
/// Deep enough that a brief scheduling hiccup does not stall generation.
const EVENT_BUFFER: usize = 64;

struct Job {
    id: u64,
    prompt: String,
    sampling: Option<Sampling>,
    max_new_tokens: Option<u32>,
    parser: ReasoningParser,
    events: mpsc::Sender<Result<Event, Error>>,
    cancelled: Arc<AtomicBool>,
}

struct Inner {
    tx: SyncSender<Job>,
    /// The request the backend is running right now.
    ///
    /// `abort` is not scoped to a run: it stops whatever is in flight on the
    /// session. If the cancelled run finished and the next one started between
    /// deciding to abort and aborting, the wrong request would die. Holding
    /// this lock across the abort, and across the worker setting the next id,
    /// means an abort either hits the right run or hits nothing.
    current: Mutex<Option<u64>>,
    backend: Arc<dyn Backend>,
    next_id: AtomicU64,
}

pub struct Worker {
    inner: Arc<Inner>,
}

/// Cancels its request when dropped.
///
/// The frontend drops its call, tonic drops the daemon's response stream, and
/// this goes with it. Nothing parses an EOF.
pub struct RunHandle {
    id: u64,
    cancelled: Arc<AtomicBool>,
    inner: Arc<Inner>,
}

impl RunHandle {
    /// Marks the request cancelled, and stops it if it is the one running.
    pub fn cancel(&self) {
        // Queued requests read this when they reach the front, so they never
        // enter prefill.
        self.cancelled.store(true, Ordering::SeqCst);

        let current = self.inner.current.lock().expect("worker lock");
        if *current == Some(self.id) {
            self.inner.backend.abort();
        }
        drop(current);
    }
}

impl Drop for RunHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl Worker {
    /// Starts the thread that owns this model's session.
    pub fn start(model: String, backend: Arc<dyn Backend>, queue_depth: usize) -> Worker {
        let (tx, rx) = sync_channel::<Job>(queue_depth);
        let inner = Arc::new(Inner {
            tx,
            current: Mutex::new(None),
            backend: backend.clone(),
            next_id: AtomicU64::new(0),
        });

        let thread_inner = inner.clone();
        std::thread::Builder::new()
            .name(format!("rkmodel-{model}"))
            .spawn(move || {
                for job in rx {
                    run_job(&thread_inner, job);
                }
                tracing::debug!(model = %model, "worker stopped");
            })
            .expect("spawning a worker thread");

        Worker { inner }
    }

    /// Queues a generation. The receiver ends when the run does.
    ///
    /// Answers `Busy` rather than blocking when the queue is full.
    pub fn submit(
        &self,
        prompt: String,
        sampling: Option<Sampling>,
        max_new_tokens: Option<u32>,
        parser: ReasoningParser,
    ) -> Result<(mpsc::Receiver<Result<Event, Error>>, RunHandle), Error> {
        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let cancelled = Arc::new(AtomicBool::new(false));
        let (events, rx) = mpsc::channel(EVENT_BUFFER);

        let job = Job {
            id,
            prompt,
            sampling,
            max_new_tokens,
            parser,
            events,
            cancelled: cancelled.clone(),
        };

        match self.inner.tx.try_send(job) {
            Ok(()) => Ok((
                rx,
                RunHandle {
                    id,
                    cancelled,
                    inner: self.inner.clone(),
                },
            )),
            Err(TrySendError::Full(_)) => Err(Error::Busy {
                retry_after_ms: BUSY_RETRY_AFTER_MS,
            }),
            Err(TrySendError::Disconnected(_)) => {
                Err(Error::Unavailable("model worker has stopped".into()))
            }
        }
    }
}

fn run_job(inner: &Arc<Inner>, mut job: Job) {
    // The client left while this waited. Drop it before it costs a prefill.
    if job.cancelled.load(Ordering::SeqCst) {
        tracing::debug!(id = job.id, "skipping a cancelled request before prefill");
        return;
    }

    *inner.current.lock().expect("worker lock") = Some(job.id);

    let mut sent_any = false;
    let result = {
        let events = job.events.clone();
        let cancelled = job.cancelled.clone();
        let parser = &mut job.parser;
        let sent_any = &mut sent_any;
        inner.backend.run(
            &job.prompt,
            job.sampling,
            job.max_new_tokens,
            &mut |piece| {
                if cancelled.load(Ordering::SeqCst) {
                    return backend::Flow::Stop;
                }
                let Some(text) = piece.text else {
                    // The runtime is holding bytes back until a UTF-8
                    // character completes. Nothing to forward.
                    return backend::Flow::Continue;
                };
                for p in parser.push(text) {
                    *sent_any = true;
                    if events.blocking_send(Ok(event_for(p))).is_err() {
                        // Nobody is listening any more.
                        return backend::Flow::Stop;
                    }
                }
                backend::Flow::Continue
            },
        )
    };

    *inner.current.lock().expect("worker lock") = None;

    if job.cancelled.load(Ordering::SeqCst) {
        tracing::debug!(id = job.id, "cancelled run produced no final event");
        return;
    }

    match result {
        Err(e) => {
            let _ = job.events.blocking_send(Err(e));
        }
        Ok(stats) => {
            for p in job.parser.finish() {
                sent_any = true;
                if job.events.blocking_send(Ok(event_for(p))).is_err() {
                    return;
                }
            }
            let _ = sent_any;
            let _ = job.events.blocking_send(Ok(Event::Done {
                finish: finish_reason(stats.generate_tokens, job.max_new_tokens),
                usage: Usage {
                    input_tokens: stats.prefill_tokens,
                    output_tokens: stats.generate_tokens,
                    reasoning_tokens: job.parser.reasoning_tokens(),
                },
            }));
        }
    }
}

fn event_for(piece: Piece) -> Event {
    match piece {
        Piece::Reasoning(s) => Event::ReasoningDelta(s),
        Piece::Text(s) => Event::TextDelta(s),
    }
}

/// RKLLM does not say why a run stopped, so hitting the budget is inferred.
fn finish_reason(generated: u32, budget: Option<u32>) -> FinishReason {
    match budget {
        Some(b) if generated >= b => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::backend::fake::FakeBackend;
    use super::*;
    use std::time::Duration;

    fn parser() -> ReasoningParser {
        ReasoningParser::disabled()
    }

    fn reasoning_parser() -> ReasoningParser {
        ReasoningParser::new("<think>", "</think>", false)
    }

    /// Drains a run to completion, returning every event.
    async fn drain(mut rx: mpsc::Receiver<Result<Event, Error>>) -> Vec<Result<Event, Error>> {
        let mut out = Vec::new();
        while let Some(e) = rx.recv().await {
            out.push(e);
        }
        out
    }

    fn deltas(events: &[Result<Event, Error>]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                Ok(Event::TextDelta(s)) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn a_run_streams_its_deltas_then_done() {
        let backend = Arc::new(FakeBackend::new(&["Hello", ", ", "world"]));
        let w = Worker::start("m".into(), backend, 4);
        let (rx, handle) = w.submit("prompt".into(), None, None, parser()).unwrap();

        let events = drain(rx).await;
        drop(handle);

        assert_eq!(deltas(&events), "Hello, world");
        match events.last() {
            Some(Ok(Event::Done { finish, usage })) => {
                assert_eq!(*finish, FinishReason::Stop);
                assert_eq!(usage.input_tokens, 7);
                assert_eq!(usage.output_tokens, 3);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_prompt_reaches_the_backend_unchanged() {
        let backend = Arc::new(FakeBackend::new(&["x"]));
        let seen = backend.seen.clone();
        let w = Worker::start("m".into(), backend, 4);
        let (rx, handle) = w
            .submit(
                "<|im_start|>user\nhi<|im_end|>\n".into(),
                None,
                None,
                parser(),
            )
            .unwrap();
        drain(rx).await;
        drop(handle);
        assert_eq!(seen.lock().unwrap()[0], "<|im_start|>user\nhi<|im_end|>\n");
    }

    #[tokio::test]
    async fn reasoning_is_split_from_content() {
        let backend = Arc::new(FakeBackend::new(&["<think>", "hmm", "</think>", "blue"]));
        let w = Worker::start("m".into(), backend, 4);
        let (rx, handle) = w
            .submit("p".into(), None, None, reasoning_parser())
            .unwrap();
        let events = drain(rx).await;
        drop(handle);

        let reasoning: String = events
            .iter()
            .filter_map(|e| match e {
                Ok(Event::ReasoningDelta(s)) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "hmm");
        assert_eq!(deltas(&events), "blue");
    }

    #[tokio::test]
    async fn hitting_the_budget_finishes_with_length() {
        let backend = Arc::new(FakeBackend::new(&["a", "b", "c", "d"]));
        let w = Worker::start("m".into(), backend, 4);
        let (rx, handle) = w.submit("p".into(), None, Some(2), parser()).unwrap();
        let events = drain(rx).await;
        drop(handle);

        assert_eq!(deltas(&events), "ab");
        match events.last() {
            Some(Ok(Event::Done { finish, .. })) => assert_eq!(*finish, FinishReason::Length),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_full_queue_is_refused_rather_than_blocking() {
        // One slow run occupies the worker, and a depth of one leaves room for
        // exactly one more.
        let backend =
            Arc::new(FakeBackend::new(&["a"]).slow(Duration::from_millis(400), Duration::ZERO));
        let w = Worker::start("m".into(), backend, 1);

        let _first = w.submit("p".into(), None, None, parser()).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _second = w.submit("p".into(), None, None, parser()).unwrap();

        let err = match w.submit("p".into(), None, None, parser()) {
            Err(e) => e,
            Ok(_) => panic!("a full queue should refuse work"),
        };
        match err {
            Error::Busy { retry_after_ms } => {
                assert!(retry_after_ms > 0);
                assert_eq!(err.http().status, 503);
            }
            other => panic!("expected Busy, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_request_cancelled_while_queued_never_reaches_the_backend() {
        let backend =
            Arc::new(FakeBackend::new(&["a"]).slow(Duration::from_millis(300), Duration::ZERO));
        let runs = backend.runs.clone();
        let w = Worker::start("m".into(), backend, 4);

        let (_rx1, first) = w.submit("p".into(), None, None, parser()).unwrap();
        let (_rx2, second) = w.submit("p".into(), None, None, parser()).unwrap();

        // The second is still waiting behind the first. Dropping its handle
        // cancels it before it ever costs a prefill.
        drop(second);
        tokio::time::sleep(Duration::from_millis(600)).await;
        drop(first);

        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "only the first run happened"
        );
    }

    #[tokio::test]
    async fn dropping_the_handle_mid_run_aborts_the_backend() {
        let backend = Arc::new(
            FakeBackend::new(&["a", "b", "c", "d", "e", "f"])
                .slow(Duration::ZERO, Duration::from_millis(60)),
        );
        let aborted = backend.aborted.clone();
        let w = Worker::start("m".into(), backend, 4);

        let (mut rx, handle) = w.submit("p".into(), None, None, parser()).unwrap();
        // Wait for generation to actually be in flight.
        let _ = rx.recv().await;
        drop(handle);

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(aborted.load(Ordering::SeqCst), "abort reached the session");
    }

    #[tokio::test]
    async fn a_cancelled_run_sends_no_done() {
        let backend = Arc::new(
            FakeBackend::new(&["a", "b", "c", "d", "e", "f"])
                .slow(Duration::ZERO, Duration::from_millis(60)),
        );
        let w = Worker::start("m".into(), backend, 4);
        let (mut rx, handle) = w.submit("p".into(), None, None, parser()).unwrap();
        let _ = rx.recv().await;
        handle.cancel();

        let mut saw_done = false;
        while let Some(event) = rx.recv().await {
            if matches!(event, Ok(Event::Done { .. })) {
                saw_done = true;
            }
        }
        assert!(!saw_done, "a cancelled run ends without a final event");
    }

    #[tokio::test]
    async fn cancelling_a_finished_run_does_not_abort_the_next_one() {
        // `abort` stops whatever is in flight, so a stale handle must not reach
        // past its own run. The lock around the current id is what prevents it.
        let backend = Arc::new(FakeBackend::new(&["a"]));
        let aborted = backend.aborted.clone();
        let w = Worker::start("m".into(), backend, 4);

        let (rx, first) = w.submit("p".into(), None, None, parser()).unwrap();
        drain(rx).await;
        // The first run is over. Cancelling it now must be a no-op.
        first.cancel();
        assert!(!aborted.load(Ordering::SeqCst));

        let (rx2, second) = w.submit("p".into(), None, None, parser()).unwrap();
        let events = drain(rx2).await;
        drop(second);
        assert_eq!(deltas(&events), "a", "the second run was untouched");
        assert!(!aborted.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_backend_failure_reaches_the_client_as_an_error() {
        let backend = Arc::new(FakeBackend::new(&["a"]).failing(Error::Runtime {
            call: "run_llm".into(),
            code: -1,
        }));
        let w = Worker::start("m".into(), backend, 4);
        let (rx, handle) = w.submit("p".into(), None, None, parser()).unwrap();
        let events = drain(rx).await;
        drop(handle);

        assert_eq!(events.len(), 1);
        match &events[0] {
            Err(Error::Runtime { .. }) => {}
            other => panic!("expected a runtime error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn runs_are_serialized_one_at_a_time() {
        let backend =
            Arc::new(FakeBackend::new(&["a"]).slow(Duration::from_millis(150), Duration::ZERO));
        let runs = backend.runs.clone();
        let w = Worker::start("m".into(), backend, 8);

        let handles: Vec<_> = (0..3)
            .map(|_| w.submit("p".into(), None, None, parser()).unwrap())
            .collect();

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            runs.load(Ordering::SeqCst) <= 2,
            "the queue serializes rather than running them together"
        );

        for (rx, handle) in handles {
            drain(rx).await;
            drop(handle);
        }
        assert_eq!(runs.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn the_budget_decides_the_finish_reason() {
        assert_eq!(finish_reason(10, Some(10)), FinishReason::Length);
        assert_eq!(finish_reason(9, Some(10)), FinishReason::Stop);
        assert_eq!(finish_reason(9999, None), FinishReason::Stop);
    }
}
