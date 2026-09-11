//! Response ids, in the shapes OpenAI clients expect.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unix seconds, which is what `created` carries.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `prefix` plus enough entropy to be unique within a process.
///
/// Clients treat these as opaque, and nothing here stores them, so a counter
/// beside the clock is enough. No two live responses share one.
pub fn generate(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}{nanos:x}{n:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_carry_their_prefix_and_do_not_repeat() {
        let a = generate("chatcmpl-");
        let b = generate("chatcmpl-");
        assert!(a.starts_with("chatcmpl-"));
        assert_ne!(a, b);
    }
}
