//! Sampling overrides and budgets.
//!
//! `InferParams::sampling` takes a whole nine-field `Sampling`, not a partial
//! override, so changing only `temperature` needs the other eight. The daemon
//! keeps each model's defaults in its config and lays the request's fields over
//! them.
//!
//! Those same nine values are given to `Param` at load, so the daemon's copy
//! and the runtime's active settings agree by construction. That matters
//! because the runtime resolves its own defaults at init and exposes only
//! `n_batch`, so there is no way to read them back. Exposing the resolved
//! `Sampling` is the change to `rkllm` that would make this copy unnecessary.

use rkmodel_server_protocol::{Error, GenerateInput};

use crate::config::Sampling;

/// What a run needs beyond the prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Budgets {
    /// `None` leaves the session's own settings alone, which is what a request
    /// carrying no sampling fields should do.
    pub sampling: Option<Sampling>,
    pub max_new_tokens: Option<u32>,
}

/// Lays a request's sampling fields over the model's configured defaults.
///
/// Returns `None` when the request set nothing, so no override is sent at all.
pub fn resolve_sampling(defaults: Sampling, request: &GenerateInput) -> Option<Sampling> {
    if request.temperature.is_none() && request.top_p.is_none() {
        return None;
    }

    let mut s = defaults;
    if let Some(top_p) = request.top_p {
        s.top_p = top_p;
    }
    if let Some(temperature) = request.temperature {
        s.temperature = temperature;
    }

    // Zero temperature means greedy. Sending it to the runtime invites a zero
    // divisor, so ask for the same behaviour the safe way.
    if s.temperature <= 0.0 {
        s.top_k = 1;
        s.temperature = 1.0;
    }
    Some(s)
}

/// The request's budget, never above the model's configured ceiling.
pub fn resolve_max_tokens(requested: Option<u32>, configured: Option<u32>) -> Option<u32> {
    match (requested, configured) {
        (Some(r), Some(c)) => Some(r.min(c)),
        (Some(r), None) => Some(r),
        (None, c) => c,
    }
}

pub fn resolve(
    defaults: Sampling,
    configured_max_new_tokens: Option<u32>,
    request: &GenerateInput,
) -> Budgets {
    Budgets {
        sampling: resolve_sampling(defaults, request),
        max_new_tokens: resolve_max_tokens(request.max_tokens, configured_max_new_tokens),
    }
}

/// A prompt too long for the model is refused rather than truncated.
///
/// Only Plan B, tokenizing in the daemon, knows the length before the run. Under
/// Plan A the runtime does this check and the daemon learns the length
/// afterwards from `PerfStat::prefill_tokens`.
pub fn check_context_length(
    prompt_tokens: u32,
    max_context_len: Option<u32>,
    max_new_tokens: Option<u32>,
) -> Result<(), Error> {
    let Some(limit) = max_context_len else {
        return Ok(());
    };
    if prompt_tokens >= limit {
        return Err(Error::ContextLengthExceeded(format!(
            "prompt is {prompt_tokens} tokens, and the model's context is {limit}"
        )));
    }
    // Leaving no room to answer is the same failure, caught earlier.
    if let Some(budget) = max_new_tokens {
        if prompt_tokens + budget > limit {
            return Err(Error::ContextLengthExceeded(format!(
                "prompt is {prompt_tokens} tokens and the budget is {budget}, \
                 which is over the model's context of {limit}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Sampling {
        Sampling {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
        }
    }

    fn request(temperature: Option<f32>, top_p: Option<f32>) -> GenerateInput {
        GenerateInput {
            temperature,
            top_p,
            ..Default::default()
        }
    }

    #[test]
    fn a_request_with_no_sampling_fields_sends_no_override() {
        assert_eq!(resolve_sampling(defaults(), &request(None, None)), None);
    }

    #[test]
    fn one_field_still_carries_the_other_eight() {
        let s = resolve_sampling(defaults(), &request(Some(0.2), None)).unwrap();
        assert_eq!(s.temperature, 0.2);
        assert_eq!(s.top_p, 0.8, "untouched fields keep the model's defaults");
        assert_eq!(s.top_k, 20);
        assert_eq!(s.mirostat_tau, 5.0);
    }

    #[test]
    fn both_fields_are_applied() {
        let s = resolve_sampling(defaults(), &request(Some(0.2), Some(0.5))).unwrap();
        assert_eq!(s.temperature, 0.2);
        assert_eq!(s.top_p, 0.5);
    }

    #[test]
    fn zero_temperature_becomes_greedy_without_a_zero_divisor() {
        let s = resolve_sampling(defaults(), &request(Some(0.0), None)).unwrap();
        assert_eq!(s.top_k, 1);
        assert!(
            s.temperature > 0.0,
            "the runtime never sees a zero temperature"
        );
    }

    #[test]
    fn a_negative_temperature_is_treated_as_greedy_too() {
        let s = resolve_sampling(defaults(), &request(Some(-1.0), None)).unwrap();
        assert_eq!(s.top_k, 1);
        assert!(s.temperature > 0.0);
    }

    #[test]
    fn the_budget_is_clamped_to_the_model_ceiling() {
        assert_eq!(resolve_max_tokens(Some(9999), Some(1024)), Some(1024));
        assert_eq!(resolve_max_tokens(Some(64), Some(1024)), Some(64));
    }

    #[test]
    fn an_absent_budget_falls_back_to_the_model_ceiling() {
        assert_eq!(resolve_max_tokens(None, Some(1024)), Some(1024));
        assert_eq!(resolve_max_tokens(None, None), None);
        assert_eq!(resolve_max_tokens(Some(64), None), Some(64));
    }

    #[test]
    fn a_prompt_over_the_context_is_refused() {
        let err = check_context_length(5000, Some(4096), None).unwrap_err();
        assert!(matches!(err, Error::ContextLengthExceeded(_)));
        assert_eq!(err.http().code, Some("context_length_exceeded"));
        assert_eq!(err.http().status, 400);
    }

    #[test]
    fn a_prompt_that_leaves_no_room_to_answer_is_refused() {
        let err = check_context_length(4000, Some(4096), Some(1024)).unwrap_err();
        assert!(matches!(err, Error::ContextLengthExceeded(_)));
    }

    #[test]
    fn a_prompt_that_fits_with_its_budget_passes() {
        assert!(check_context_length(1000, Some(4096), Some(1024)).is_ok());
    }

    #[test]
    fn no_configured_context_means_no_check_here() {
        assert!(check_context_length(999_999, None, Some(1024)).is_ok());
    }
}
