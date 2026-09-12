//! The backend that actually runs a model, through `librkllmrt`.
//!
//! Symbols are resolved by `dlopen` at run time, so this compiles anywhere and
//! only needs the runtime present where it runs.

use std::sync::Mutex;

use anyhow::{Context, Result};
use rkllm::rkllm_sys::RkllmRuntime;
use rkllm::{CallState, Control, InferParams, Input, Param, RkllmSession, SessionBuilder};
use rkmodel_server_protocol::Error;

use crate::config::{ModelConfig, Sampling};
use crate::worker::backend::{Backend, Flow, Piece, Prompt, RunStats};

pub struct RkllmBackend {
    session: RkllmSession<RkllmRuntime>,
    model: String,
    /// The last prompt this session prefilled, and how many tokens it took.
    ///
    /// Measured on the board: when two runs in a row carry the identical
    /// prompt, the runtime reuses its KV cache, skips prefill entirely, and
    /// reports `prefill_tokens = 0` with `prefill_time_ms = 0`. Taking that at
    /// face value would report zero input tokens for a repeated request, so the
    /// real count is remembered and substituted. A different prompt in between
    /// evicts the cache and prefill is measured again.
    last_prefill: Mutex<Option<(String, u32)>>,
    /// Clear the KV cache before each run. See `ModelConfig::reuse_kv_cache`.
    clear_cache: bool,
}

/// The input token count to report, given what the runtime said.
///
/// A zero from a skipped prefill is only trustworthy as "the same prompt as
/// last time", so it is replaced only when the prompt actually matches.
fn prefill_tokens(last: Option<&(String, u32)>, prompt: &str, reported: u32) -> u32 {
    if reported > 0 {
        return reported;
    }
    match last {
        Some((previous, tokens)) if previous == prompt => *tokens,
        _ => reported,
    }
}

/// The runtime's own turn framing is switched off, so the prompt text reaches
/// the model exactly as the daemon rendered it from the model's chat template.
/// This is Plan A in the design. A model the daemon tokenizes itself, Plan B,
/// passes `Input::tokens` instead, and the template setting is moot for it.
const NO_RUNTIME_TEMPLATE: (&str, &str, &str) = ("", "", "");

impl RkllmBackend {
    pub fn load(model: &ModelConfig) -> Result<RkllmBackend> {
        let weights = model
            .rkllm
            .as_deref()
            .with_context(|| format!("model {} has no rkllm path", model.id))?;

        let mut param = Param::new(weights.as_os_str().as_encoded_bytes().to_vec())
            .with_context(|| format!("building params for {}", model.id))?;

        // The same nine values the daemon lays request overrides on top of, so
        // its copy and the runtime's active settings agree by construction.
        let s = model.sampling;
        param = param
            .temperature(s.temperature)
            .top_p(s.top_p)
            .top_k(s.top_k as i32)
            .repeat_penalty(s.repeat_penalty)
            .frequency_penalty(s.frequency_penalty)
            .presence_penalty(s.presence_penalty)
            .mirostat(s.mirostat as i32)
            .mirostat_tau(s.mirostat_tau)
            .mirostat_eta(s.mirostat_eta);

        if let Some(n) = model.max_context_len {
            param = param.max_context_len(n as i32);
        }
        if let Some(n) = model.max_new_tokens {
            param = param.max_new_tokens(n as i32);
        }

        // The toolkit quantizes the layers to w8a8 but leaves the embedding
        // tables at fp16, so a model's embeddings can outweigh its weights
        // several times over. Reading them from flash keeps that region out of
        // resident memory, at the cost of a read per token.
        if let Some(from_flash) = model.embed_flash {
            param = param.embed_flash(from_flash);
        }

        // By default the runtime leaves special tokens out of callback text.
        // Gemma's reasoning and tool markers are special tokens, so a parser
        // would never see them. Kept, every token arrives as its text on a
        // callback of its own, and the worker drops the ones that end a turn.
        param = param.skip_special_token(false);

        // Streaming only works from a single-input session, and a batched one
        // needs exactly n_batch inputs per run. Batching is future work.
        param = param
            .n_batch(1)
            .with_context(|| format!("setting n_batch for {}", model.id))?;

        let library = rkllm::find_library_path()
            .next()
            .unwrap_or_else(|| rkllm::LIBRARY_NAME.into());
        tracing::info!(model = %model.id, library = %library.display(), "loading weights");

        let session = SessionBuilder::new(&param)
            .open_library(&library)
            .with_context(|| {
                format!(
                    "loading {} through {}",
                    weights.display(),
                    library.display()
                )
            })?;

        session
            .set_chat_template(
                NO_RUNTIME_TEMPLATE.0,
                NO_RUNTIME_TEMPLATE.1,
                NO_RUNTIME_TEMPLATE.2,
            )
            .with_context(|| format!("clearing the runtime template for {}", model.id))?;

        Ok(RkllmBackend {
            session,
            model: model.id.clone(),
            last_prefill: Mutex::new(None),
            clear_cache: model.reuse_kv_cache == Some(false),
        })
    }
}

fn to_rkllm_sampling(s: Sampling) -> rkllm::Sampling {
    rkllm::Sampling {
        top_k: s.top_k as i32,
        top_p: s.top_p,
        temperature: s.temperature,
        repeat_penalty: s.repeat_penalty,
        frequency_penalty: s.frequency_penalty,
        presence_penalty: s.presence_penalty,
        mirostat: s.mirostat as i32,
        mirostat_tau: s.mirostat_tau,
        mirostat_eta: s.mirostat_eta,
    }
}

impl Backend for RkllmBackend {
    fn run(
        &self,
        prompt: &Prompt,
        sampling: Option<Sampling>,
        max_new_tokens: Option<u32>,
        on_piece: &mut dyn FnMut(Piece<'_>) -> Flow,
    ) -> Result<RunStats, Error> {
        let mut input = match &prompt.tokens {
            // The runtime's own tokenizer splits some models' prompts
            // differently from Hugging Face's, and Gemma 4 behaves worse for it.
            Some(tokens) => Input::tokens(tokens.clone()),
            None => Input::prompt(prompt.text.as_bytes().to_vec()).map_err(|e| Error::Runtime {
                call: format!("Input::prompt: {e}"),
                code: -1,
            })?,
        };

        // The OpenAI APIs are stateless, so every run renders the whole
        // transcript and the runtime's own history is never used.
        let mut params = InferParams::new().keep_history(false);
        if let Some(s) = sampling {
            params = params.sampling(to_rkllm_sampling(s));
        }
        if let Some(n) = max_new_tokens {
            params = params.max_new_tokens(n as i32);
        }

        if self.clear_cache {
            self.session
                .clear_kv_cache(false)
                .map_err(|e| Error::Runtime {
                    call: format!("clear_kv_cache on {}: {e}", self.model),
                    code: -1,
                })?;
        }

        let mut stats = RunStats::default();
        let mut failed = false;
        let mut finishes = 0u32;

        self.session
            .run_llm(&mut input, &params, |output| match output.state() {
                CallState::Error => {
                    failed = true;
                    Control::Pause
                }
                CallState::Finish => {
                    finishes += 1;
                    match output.perf() {
                        Some(perf) => {
                            tracing::debug!(
                                model = %self.model,
                                finishes,
                                prefill_tokens = perf.prefill_tokens,
                                generate_tokens = perf.generate_tokens,
                                prefill_ms = perf.prefill_time_ms,
                                generate_ms = perf.generate_time_ms,
                                "final callback"
                            );
                            stats.prefill_tokens = perf.prefill_tokens.max(0) as u32;
                            stats.generate_tokens = perf.generate_tokens.max(0) as u32;
                        }
                        None => tracing::debug!(
                            model = %self.model,
                            finishes,
                            "final callback carried no perf stats"
                        ),
                    }
                    Control::Continue
                }
                // `Waiting` means the runtime is holding bytes back until a
                // UTF-8 character completes, and carries no text.
                _ => match on_piece(Piece {
                    text: output.text(),
                }) {
                    Flow::Continue => Control::Continue,
                    Flow::Stop => Control::Pause,
                },
            })
            .map_err(|e| Error::Runtime {
                call: format!("run_llm on {}: {e}", self.model),
                code: -1,
            })?;

        if failed {
            return Err(Error::Runtime {
                call: format!("run_llm on {} reported an error state", self.model),
                code: -1,
            });
        }

        // Under Plan B the prompt's length is known exactly, and the runtime's
        // count is not: measured on the board, a Gemma prompt of 116 tokens
        // that shared a prefix with the run before it reported 35, which is
        // what it prefilled rather than what the prompt is. The exact length
        // also covers a run stopped from a callback before anything was
        // reported.
        if let Some(tokens) = &prompt.tokens {
            stats.prefill_tokens = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        }

        let mut last = self.last_prefill.lock().expect("prefill lock");
        stats.prefill_tokens = prefill_tokens(last.as_ref(), &prompt.text, stats.prefill_tokens);
        if stats.prefill_tokens > 0 {
            *last = Some((prompt.text.clone(), stats.prefill_tokens));
        }
        drop(last);

        Ok(stats)
    }

    fn abort(&self) {
        // Step 5 of the cancellation path only fires on the next chunk, and the
        // first chunk comes after prefill, which runs for seconds on a long
        // prompt. So the worker aborts rather than waiting for a callback.
        if let Err(e) = self.session.abort() {
            tracing::warn!(model = %self.model, "abort failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::prefill_tokens;

    #[test]
    fn a_measured_prefill_is_reported_as_is() {
        assert_eq!(prefill_tokens(None, "hello", 12), 12);
    }

    #[test]
    fn a_skipped_prefill_on_the_same_prompt_reuses_the_real_count() {
        let last = ("hello".to_string(), 12);
        assert_eq!(prefill_tokens(Some(&last), "hello", 0), 12);
    }

    #[test]
    fn a_zero_on_a_different_prompt_is_left_alone() {
        // Not a cache hit, so there is nothing honest to substitute.
        let last = ("hello".to_string(), 12);
        assert_eq!(prefill_tokens(Some(&last), "goodbye", 0), 0);
    }

    #[test]
    fn a_fresh_session_has_nothing_to_reuse() {
        assert_eq!(prefill_tokens(None, "hello", 0), 0);
    }
}
