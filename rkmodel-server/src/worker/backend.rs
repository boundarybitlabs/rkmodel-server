//! The seam between the worker and the closed-source runtime.
//!
//! Everything above this trait is ordinary Rust with tests. Everything below it
//! is `librkllmrt`, which can only run on the board. Keeping the line here is
//! what lets the queue and cancellation logic be tested on any host.

use rkmodel_server_protocol::Error;

use crate::config::Sampling;

/// What the worker tells the backend to do next, from inside a callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// Stop this run and return. The runtime has no separate stop code, so this
    /// becomes `Control::Pause`.
    Stop,
}

/// What a run is given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// The rendered prompt. Always present, since the reasoning parser and the
    /// repeated-prompt accounting both read it.
    pub text: String,
    /// The prompt as token ids, for a model the daemon tokenizes itself. The
    /// runtime is then handed these rather than the text.
    pub tokens: Option<Vec<i32>>,
}

impl From<String> for Prompt {
    fn from(text: String) -> Prompt {
        Prompt { text, tokens: None }
    }
}

impl From<&str> for Prompt {
    fn from(text: &str) -> Prompt {
        Prompt::from(text.to_string())
    }
}

/// One callback's worth of output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece<'a> {
    /// `None` when the runtime held bytes back waiting for a UTF-8 character to
    /// complete, which is how a delta never splits one.
    pub text: Option<&'a str>,
}

/// What the final callback reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    pub prefill_tokens: u32,
    pub generate_tokens: u32,
}

/// A loaded model, ready to run one generation at a time.
pub trait Backend: Send + Sync {
    /// Runs one generation to completion, or until `on_piece` says stop.
    ///
    /// Blocking. The worker calls this on its own dedicated thread.
    fn run(
        &self,
        prompt: &Prompt,
        sampling: Option<Sampling>,
        max_new_tokens: Option<u32>,
        on_piece: &mut dyn FnMut(Piece<'_>) -> Flow,
    ) -> Result<RunStats, Error>;

    /// Stops whatever is in flight on this session.
    ///
    /// Not scoped to a run, which is why the worker holds a lock around the
    /// identity of the request it is running.
    fn abort(&self);
}

#[cfg(test)]
pub mod fake {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Replays scripted chunks, and records what was asked of it.
    pub struct FakeBackend {
        pub chunks: Vec<String>,
        /// Slept before the first chunk, standing in for prefill.
        pub prefill: Duration,
        /// Slept between chunks, so a test can cancel mid-run.
        pub between: Duration,
        pub aborted: Arc<AtomicBool>,
        pub runs: Arc<AtomicUsize>,
        pub stats: RunStats,
        pub seen: Arc<Mutex<Vec<String>>>,
        pub fail_with: Option<Error>,
    }

    impl FakeBackend {
        pub fn new(chunks: &[&str]) -> FakeBackend {
            FakeBackend {
                chunks: chunks.iter().map(|s| s.to_string()).collect(),
                prefill: Duration::ZERO,
                between: Duration::ZERO,
                aborted: Arc::new(AtomicBool::new(false)),
                runs: Arc::new(AtomicUsize::new(0)),
                stats: RunStats {
                    prefill_tokens: 7,
                    generate_tokens: chunks.len() as u32,
                },
                seen: Arc::new(Mutex::new(Vec::new())),
                fail_with: None,
            }
        }

        pub fn slow(mut self, prefill: Duration, between: Duration) -> Self {
            self.prefill = prefill;
            self.between = between;
            self
        }

        pub fn failing(mut self, error: Error) -> Self {
            self.fail_with = Some(error);
            self
        }
    }

    impl Backend for FakeBackend {
        fn run(
            &self,
            prompt: &Prompt,
            _sampling: Option<Sampling>,
            max_new_tokens: Option<u32>,
            on_piece: &mut dyn FnMut(Piece<'_>) -> Flow,
        ) -> Result<RunStats, Error> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            self.seen.lock().unwrap().push(prompt.text.clone());
            if let Some(e) = &self.fail_with {
                return Err(e.clone());
            }

            std::thread::sleep(self.prefill);
            let budget = max_new_tokens.unwrap_or(u32::MAX) as usize;
            let mut sent = 0u32;
            for chunk in self.chunks.iter().take(budget) {
                if on_piece(Piece { text: Some(chunk) }) == Flow::Stop {
                    return Ok(RunStats {
                        prefill_tokens: self.stats.prefill_tokens,
                        generate_tokens: sent,
                    });
                }
                sent += 1;
                std::thread::sleep(self.between);
            }
            Ok(RunStats {
                prefill_tokens: self.stats.prefill_tokens,
                generate_tokens: sent,
            })
        }

        fn abort(&self) {
            self.aborted.store(true, Ordering::SeqCst);
        }
    }
}
