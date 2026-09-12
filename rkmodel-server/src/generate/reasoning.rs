//! Splitting a model's output into reasoning and content.
//!
//! Models that reason are configured with the markers around their reasoning.
//! The daemon splits the stream, and the frontend returns the reasoning as its
//! own field.

/// One piece of split output. A chunk can produce several, when a marker sits
/// in the middle of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Reasoning(String),
    Text(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Text,
    Reasoning,
}

pub struct ReasoningParser {
    markers: Option<Markers>,
    state: State,
    /// A tail that could still turn out to be the start of a marker, such as
    /// `<thi`. Held until the next chunk settles it.
    held: String,
    reasoning_tokens: u32,
}

#[derive(Debug, Clone)]
struct Markers {
    start: String,
    end: String,
}

impl ReasoningParser {
    /// For a model with no reasoning markers. Everything is content.
    pub fn disabled() -> ReasoningParser {
        ReasoningParser {
            markers: None,
            state: State::Text,
            held: String::new(),
            reasoning_tokens: 0,
        }
    }

    /// `starts_in_reasoning` is for templates that end the prompt with the
    /// start marker themselves. The model's output then begins inside
    /// reasoning, and only the closing marker ever appears.
    pub fn new(start: &str, end: &str, starts_in_reasoning: bool) -> ReasoningParser {
        ReasoningParser {
            markers: Some(Markers {
                start: start.to_string(),
                end: end.to_string(),
            }),
            state: if starts_in_reasoning {
                State::Reasoning
            } else {
                State::Text
            },
            held: String::new(),
            reasoning_tokens: 0,
        }
    }

    /// Whether a rendered prompt leaves the model already inside its reasoning
    /// block.
    ///
    /// Both sides are trimmed, since a start marker can end in a newline, as
    /// Gemma's `<|channel>thought\n` does.
    pub fn prompt_opens_reasoning(prompt: &str, start: &str) -> bool {
        prompt.trim_end().ends_with(start.trim_end())
    }

    /// Counts callbacks that arrived while inside reasoning, which assumes one
    /// callback per generated token.
    pub fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    pub fn in_reasoning(&self) -> bool {
        self.state == State::Reasoning
    }

    /// Feeds one chunk of model output through the splitter.
    pub fn push(&mut self, chunk: &str) -> Vec<Piece> {
        if self.state == State::Reasoning {
            self.reasoning_tokens += 1;
        }

        let Some(markers) = self.markers.clone() else {
            return emit(State::Text, chunk);
        };

        let mut buf = std::mem::take(&mut self.held);
        buf.push_str(chunk);

        let mut out = Vec::new();
        loop {
            let marker = match self.state {
                State::Text => &markers.start,
                State::Reasoning => &markers.end,
            };

            if let Some(at) = buf.find(marker.as_str()) {
                out.extend(emit(self.state, &buf[..at]));
                buf = buf[at + marker.len()..].to_string();
                self.state = match self.state {
                    State::Text => State::Reasoning,
                    State::Reasoning => State::Text,
                };
                continue;
            }

            // No whole marker. Anything that could still become one waits for
            // the next chunk; the rest is safe to emit now.
            let keep = partial_suffix(&buf, marker);
            let split = buf.len() - keep;
            out.extend(emit(self.state, &buf[..split]));
            self.held = buf[split..].to_string();
            break;
        }
        out
    }

    /// Flushes anything held back. A tail that never completed a marker was
    /// ordinary text all along.
    pub fn finish(&mut self) -> Vec<Piece> {
        let held = std::mem::take(&mut self.held);
        emit(self.state, &held)
    }
}

fn emit(state: State, text: &str) -> Vec<Piece> {
    if text.is_empty() {
        return Vec::new();
    }
    match state {
        State::Text => vec![Piece::Text(text.to_string())],
        State::Reasoning => vec![Piece::Reasoning(text.to_string())],
    }
}

/// How many bytes at the end of `buf` form a proper prefix of `marker`.
///
/// Longest first, so `<think` beats `<` and the held tail stays minimal.
fn partial_suffix(buf: &str, marker: &str) -> usize {
    let max = marker.len().saturating_sub(1).min(buf.len());
    for len in (1..=max).rev() {
        let at = buf.len() - len;
        // A marker boundary can land mid-character, and slicing there panics.
        if !buf.is_char_boundary(at) {
            continue;
        }
        if marker.starts_with(&buf[at..]) {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> ReasoningParser {
        ReasoningParser::new("<think>", "</think>", false)
    }

    /// Feeds every chunk and returns everything emitted, flush included.
    fn run(p: &mut ReasoningParser, chunks: &[&str]) -> Vec<Piece> {
        let mut out = Vec::new();
        for c in chunks {
            out.extend(p.push(c));
        }
        out.extend(p.finish());
        out
    }

    fn text(s: &str) -> Piece {
        Piece::Text(s.into())
    }
    fn reasoning(s: &str) -> Piece {
        Piece::Reasoning(s.into())
    }

    #[test]
    fn output_with_no_markers_is_all_content() {
        let mut p = parser();
        assert_eq!(
            run(&mut p, &["the sky ", "is blue"]),
            vec![text("the sky "), text("is blue")]
        );
        assert_eq!(p.reasoning_tokens(), 0);
    }

    #[test]
    fn a_whole_block_in_one_chunk_splits() {
        let mut p = parser();
        assert_eq!(
            run(&mut p, &["<think>hmm</think>blue"]),
            vec![reasoning("hmm"), text("blue")]
        );
    }

    #[test]
    fn a_marker_split_across_chunks_is_held_until_it_settles() {
        let mut p = parser();
        // The `<thi` must not reach the client as content.
        assert_eq!(p.push("<thi"), vec![]);
        assert_eq!(p.push("nk>hmm"), vec![reasoning("hmm")]);
        assert_eq!(p.push("</thi"), vec![]);
        assert_eq!(p.push("nk>blue"), vec![text("blue")]);
    }

    #[test]
    fn a_partial_that_was_never_a_marker_is_released_as_text() {
        let mut p = parser();
        assert_eq!(p.push("<thi"), vec![]);
        assert_eq!(p.push("s is fine"), vec![text("<this is fine")]);
    }

    #[test]
    fn a_partial_at_the_very_end_is_flushed_by_finish() {
        let mut p = parser();
        assert_eq!(p.push("done<thi"), vec![text("done")]);
        assert_eq!(p.finish(), vec![text("<thi")]);
    }

    #[test]
    fn a_template_that_opens_the_block_starts_in_reasoning() {
        // Qwen3's template ends the prompt with `<think>`, so only the closing
        // marker ever arrives.
        let mut p = ReasoningParser::new("<think>", "</think>", true);
        assert_eq!(
            run(&mut p, &["hmm", "</think>", "blue"]),
            vec![reasoning("hmm"), text("blue")]
        );
    }

    #[test]
    fn a_start_marker_ending_in_a_newline_is_recognised() {
        assert!(ReasoningParser::prompt_opens_reasoning(
            "<tool_response|><|channel>thought\n",
            "<|channel>thought\n"
        ));
    }

    #[test]
    fn a_prompt_ending_in_the_start_marker_is_recognised() {
        assert!(ReasoningParser::prompt_opens_reasoning(
            "<|im_start|>assistant\n<think>\n",
            "<think>"
        ));
        assert!(!ReasoningParser::prompt_opens_reasoning(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<think>"
        ));
    }

    #[test]
    fn reasoning_tokens_count_callbacks_inside_the_block() {
        let mut p = ReasoningParser::new("<think>", "</think>", true);
        p.push("a");
        p.push("b");
        p.push("c</think>");
        // The fourth callback arrives after the block closed.
        p.push("blue");
        assert_eq!(p.reasoning_tokens(), 3);
    }

    #[test]
    fn several_blocks_in_one_chunk_all_split() {
        let mut p = parser();
        assert_eq!(
            run(&mut p, &["a<think>b</think>c<think>d</think>e"]),
            vec![
                text("a"),
                reasoning("b"),
                text("c"),
                reasoning("d"),
                text("e"),
            ]
        );
    }

    #[test]
    fn a_disabled_parser_passes_everything_through() {
        let mut p = ReasoningParser::disabled();
        assert_eq!(
            run(&mut p, &["<think>", "not a marker here"]),
            vec![text("<think>"), text("not a marker here")]
        );
        assert_eq!(p.reasoning_tokens(), 0);
    }

    #[test]
    fn a_multibyte_character_before_a_partial_marker_does_not_panic() {
        // Scanning back for a marker prefix steps through byte positions that
        // land inside the snowman. Those must be skipped, not sliced.
        let mut p = parser();
        assert_eq!(p.push("\u{2603}<thi"), vec![text("\u{2603}")]);
        assert_eq!(p.push("nk>hmm</think>x"), vec![reasoning("hmm"), text("x")]);
    }

    #[test]
    fn a_held_tail_may_end_mid_character_worth_of_lookback() {
        // `partial_suffix` must not report a length that starts inside a
        // character, however far back it looks.
        assert_eq!(partial_suffix("\u{2603}<", "<think>"), 1);
        assert_eq!(partial_suffix("a\u{2603}", "<think>"), 0);
    }

    #[test]
    fn the_longest_partial_suffix_wins() {
        assert_eq!(partial_suffix("ab<thin", "<think>"), 5);
        assert_eq!(partial_suffix("ab<", "<think>"), 1);
        assert_eq!(partial_suffix("nothing", "<think>"), 0);
        // A whole marker is not a partial; the caller finds it first.
        assert_eq!(partial_suffix("<think>", "<think>"), 0);
    }
}
