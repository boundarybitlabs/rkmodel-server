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
use crate::generate::output::OutputParser;

pub use backend::{Backend, Prompt};

/// What a client is told to wait before retrying a full queue. The daemon
/// cannot know how long the run in front will take, so this is a floor that
/// keeps a rejected client from spinning.
const BUSY_RETRY_AFTER_MS: u32 = 1_000;

/// How many events may sit between the worker thread and the connection.
/// Deep enough that a brief scheduling hiccup does not stall generation.
const EVENT_BUFFER: usize = 64;

struct Job {
    id: u64,
    prompt: Prompt,
    sampling: Option<Sampling>,
    max_new_tokens: Option<u32>,
    parser: OutputParser,
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
    /// Tokens that end a turn, which arrive as text because the runtime is told
    /// to keep special tokens, and which a client must never see.
    end_of_turn: Vec<String>,
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
    ///
    /// `end_of_turn` lists the tokens dropped from the output, such as Qwen's
    /// `<|im_end|>` or Gemma's `<turn|>`.
    pub fn start(
        model: String,
        backend: Arc<dyn Backend>,
        queue_depth: usize,
        end_of_turn: Vec<String>,
    ) -> Worker {
        let (tx, rx) = sync_channel::<Job>(queue_depth);
        let inner = Arc::new(Inner {
            tx,
            current: Mutex::new(None),
            backend: backend.clone(),
            next_id: AtomicU64::new(0),
            end_of_turn,
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
        prompt: Prompt,
        sampling: Option<Sampling>,
        max_new_tokens: Option<u32>,
        parser: OutputParser,
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

    // Counted here rather than taken from the runtime. Measured on the board:
    // when a run does its own prefill, `generate_tokens` can come back one
    // below the number of callbacks that actually carried text, which would
    // report a truncated answer as a natural stop. See `generated`.
    let mut text_callbacks = 0u32;
    let result = {
        let events = job.events.clone();
        let cancelled = job.cancelled.clone();
        let parser = &mut job.parser;
        let text_callbacks = &mut text_callbacks;
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
                // Each special token arrives on a callback of its own, so an
                // end-of-turn token is the whole of its text. It is not counted
                // either, since the runtime's `generate_tokens` leaves it out.
                if inner.end_of_turn.iter().any(|t| t == text) {
                    return backend::Flow::Continue;
                }
                *text_callbacks += 1;
                let (out, stop) = parser.push(text);
                for event in out {
                    if events.blocking_send(Ok(event)).is_err() {
                        // Nobody is listening any more.
                        return backend::Flow::Stop;
                    }
                }
                // A parser stops a run that would otherwise go on to invent a
                // tool's result, or make a second call it was told not to.
                if stop {
                    backend::Flow::Stop
                } else {
                    backend::Flow::Continue
                }
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
            for event in job.parser.finish() {
                if job.events.blocking_send(Ok(event)).is_err() {
                    return;
                }
            }
            let generated = generated_tokens(stats.generate_tokens, text_callbacks);
            let finish = if job.parser.called() {
                FinishReason::ToolCalls
            } else {
                finish_reason(generated, job.max_new_tokens)
            };
            let _ = job.events.blocking_send(Ok(Event::Done {
                finish,
                usage: Usage {
                    input_tokens: stats.prefill_tokens,
                    output_tokens: generated,
                    reasoning_tokens: job.parser.reasoning_tokens(),
                },
            }));
        }
    }
}

/// How many tokens this run produced, reconciling two counts that disagree.
///
/// The runtime's `generate_tokens` is normally right, and is the only one that
/// sees tokens which produced no text, such as an end-of-sequence token or one
/// held back mid-character. But on the board it has also come back one below
/// the number of callbacks that carried text, which would both under-report
/// usage and hide a run that was cut off by its budget. Taking the larger keeps
/// the reported count consistent with what the client was actually sent.
fn generated_tokens(reported: u32, text_callbacks: u32) -> u32 {
    reported.max(text_callbacks)
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
    use crate::config::ToolFormat;
    use crate::generate::reasoning::ReasoningParser;
    use crate::generate::tools::ToolCallParser;
    use std::time::Duration;

    fn parser() -> OutputParser {
        OutputParser::plain()
    }

    fn reasoning_parser() -> OutputParser {
        OutputParser::new(ReasoningParser::new("<think>", "</think>", false), None)
    }

    fn tool_parser(stop_after_first: bool) -> OutputParser {
        OutputParser::new(
            ReasoningParser::disabled(),
            Some(ToolCallParser::new(
                ToolFormat::Gemma4,
                vec!["get_weather".into()],
                stop_after_first,
            )),
        )
    }

    #[tokio::test]
    async fn a_run_that_calls_a_tool_finishes_tool_calls_and_stops_before_a_result() {
        let backend = Arc::new(FakeBackend::new(&[
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
            "<|tool_response>",
            "response:get_weather{value:30}",
            "<tool_response|>",
        ]));
        let w = Worker::start("m".into(), backend, 4, Vec::new());
        let (rx, handle) = w
            .submit("p".into(), None, None, tool_parser(false))
            .unwrap();
        let events = drain(rx).await;
        drop(handle);

        assert!(
            matches!(&events[0], Ok(Event::ToolCall(c)) if c.name == "get_weather"),
            "{events:?}"
        );
        let Some(Ok(Event::Done { finish, usage })) = events.last() else {
            panic!("no final event: {events:?}");
        };
        assert_eq!(*finish, FinishReason::ToolCalls);
        // The fake counts what it delivered before being told to stop: the
        // call and the start of the invented result, and nothing after.
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(events.len(), 2, "{events:?}");
    }

    #[tokio::test]
    async fn a_call_cut_off_by_the_budget_is_text_and_length() {
        let backend = Arc::new(FakeBackend::new(&["<|tool_call>call:get_", "weather{"]));
        let w = Worker::start("m".into(), backend, 4, Vec::new());
        let (rx, handle) = w
            .submit("p".into(), None, Some(2), tool_parser(false))
            .unwrap();
        let events = drain(rx).await;
        drop(handle);
        assert_eq!(deltas(&events), "<|tool_call>call:get_weather{");
        assert!(matches!(
            events.last(),
            Some(Ok(Event::Done {
                finish: FinishReason::Length,
                ..
            }))
        ));
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
    async fn end_of_turn_tokens_are_dropped_and_not_counted() {
        let backend = Arc::new(FakeBackend::new(&["Paris", "<turn|>"]));
        let w = Worker::start(
            "m".into(),
            backend,
            4,
            vec!["<eos>".into(), "<turn|>".into()],
        );
        let (rx, handle) = w.submit("p".into(), None, None, parser()).unwrap();
        let events = drain(rx).await;
        drop(handle);

        assert_eq!(deltas(&events), "Paris");
        // The fake reports two generated tokens, as a runtime counting the end
        // token would. The daemon's own count, one, must not exceed it.
        let Some(Ok(Event::Done { usage, .. })) = events.last() else {
            panic!("no final event: {events:?}");
        };
        assert_eq!(usage.output_tokens, 2);
    }

    #[tokio::test]
    async fn text_that_merely_contains_an_end_token_is_kept() {
        let backend = Arc::new(FakeBackend::new(&["say <turn|> please"]));
        let w = Worker::start("m".into(), backend, 4, vec!["<turn|>".into()]);
        let (rx, handle) = w.submit("p".into(), None, None, parser()).unwrap();
        let events = drain(rx).await;
        drop(handle);
        assert_eq!(deltas(&events), "say <turn|> please");
    }

    #[tokio::test]
    async fn a_run_streams_its_deltas_then_done() {
        let backend = Arc::new(FakeBackend::new(&["Hello", ", ", "world"]));
        let w = Worker::start("m".into(), backend, 4, Vec::new());
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
        let w = Worker::start("m".into(), backend, 4, Vec::new());
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
        let w = Worker::start("m".into(), backend, 4, Vec::new());
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
        let w = Worker::start("m".into(), backend, 4, Vec::new());
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
        let w = Worker::start("m".into(), backend, 1, Vec::new());

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
        let w = Worker::start("m".into(), backend, 4, Vec::new());

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
        let w = Worker::start("m".into(), backend, 4, Vec::new());

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
        let w = Worker::start("m".into(), backend, 4, Vec::new());
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
        let w = Worker::start("m".into(), backend, 4, Vec::new());

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
        let w = Worker::start("m".into(), backend, 4, Vec::new());
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
        let w = Worker::start("m".into(), backend, 8, Vec::new());

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

    #[tokio::test]
    async fn a_runtime_that_under_reports_still_shows_length() {
        // The board does this: ten callbacks carried text, and the runtime
        // reported nine tokens against a budget of ten. Reporting `stop` would
        // tell a client the model finished when it was cut off.
        let mut backend = FakeBackend::new(&["a", "b", "c", "d", "e"]);
        backend.stats.generate_tokens = 4;
        let w = Worker::start("m".into(), Arc::new(backend), 4, Vec::new());
        let (rx, handle) = w.submit("p".into(), None, Some(5), parser()).unwrap();
        let events = drain(rx).await;
        drop(handle);

        assert_eq!(deltas(&events), "abcde");
        match events.last() {
            Some(Ok(Event::Done { finish, usage })) => {
                assert_eq!(*finish, FinishReason::Length);
                assert_eq!(usage.output_tokens, 5, "usage matches what was sent");
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn the_larger_of_the_two_counts_wins() {
        // The runtime sees tokens that carried no text, so it can legitimately
        // report more than the callbacks did.
        assert_eq!(generated_tokens(12, 10), 12);
        // And it has been seen reporting one fewer than it delivered.
        assert_eq!(generated_tokens(9, 10), 10);
        assert_eq!(generated_tokens(0, 0), 0);
    }

    #[test]
    fn the_budget_decides_the_finish_reason() {
        assert_eq!(finish_reason(10, Some(10)), FinishReason::Length);
        assert_eq!(finish_reason(9, Some(10)), FinishReason::Stop);
        assert_eq!(finish_reason(9999, None), FinishReason::Stop);
    }
}
