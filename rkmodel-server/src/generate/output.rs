//! A model's output, split into the events a client sees.
//!
//! Reasoning is split off first, and only what is left is searched for tool
//! calls, so a call a model drafts inside its reasoning is never taken for one.

use rkmodel_server_protocol::Event;

use crate::generate::reasoning::{Piece, ReasoningParser};
use crate::generate::tools::{ToolCallParser, ToolPiece};

pub struct OutputParser {
    reasoning: ReasoningParser,
    tools: Option<ToolCallParser>,
}

impl OutputParser {
    /// `tools` is `None` for a run that offered none.
    pub fn new(reasoning: ReasoningParser, tools: Option<ToolCallParser>) -> Self {
        OutputParser { reasoning, tools }
    }

    /// Everything is content.
    pub fn plain() -> Self {
        OutputParser::new(ReasoningParser::disabled(), None)
    }

    /// Feeds one chunk. The flag is true when the run should stop.
    pub fn push(&mut self, chunk: &str) -> (Vec<Event>, bool) {
        let pieces = self.reasoning.push(chunk);
        self.route(pieces)
    }

    /// Flushes both parsers, reasoning first, since its held tail comes before
    /// anything the tool parser holds.
    pub fn finish(&mut self) -> Vec<Event> {
        let pieces = self.reasoning.finish();
        let (mut events, _) = self.route(pieces);
        if let Some(tools) = &mut self.tools {
            events.extend(tools.finish().into_iter().map(event_for));
        }
        events
    }

    fn route(&mut self, pieces: Vec<Piece>) -> (Vec<Event>, bool) {
        let mut events = Vec::new();
        let mut stop = false;
        for piece in pieces {
            match (piece, &mut self.tools) {
                (Piece::Reasoning(s), _) => events.push(Event::ReasoningDelta(s)),
                (Piece::Text(s), None) => events.push(Event::TextDelta(s)),
                (Piece::Text(s), Some(tools)) => {
                    let (out, stopped) = tools.push(&s);
                    events.extend(out.into_iter().map(event_for));
                    if stopped {
                        stop = true;
                        break;
                    }
                }
            }
        }
        (events, stop)
    }

    pub fn reasoning_tokens(&self) -> u32 {
        self.reasoning.reasoning_tokens()
    }

    /// Whether the run emitted a call, which makes its finish `ToolCalls`.
    pub fn called(&self) -> bool {
        self.tools.as_ref().is_some_and(|t| t.calls() > 0)
    }
}

fn event_for(piece: ToolPiece) -> Event {
    match piece {
        ToolPiece::Text(s) => Event::TextDelta(s),
        ToolPiece::Call(c) => Event::ToolCall(c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ToolFormat;

    fn run(p: &mut OutputParser, chunks: &[&str]) -> Vec<Event> {
        let mut out = Vec::new();
        for c in chunks {
            let (events, stop) = p.push(c);
            out.extend(events);
            if stop {
                break;
            }
        }
        out.extend(p.finish());
        out
    }

    #[test]
    fn a_call_drafted_inside_reasoning_is_not_a_call() {
        let mut p = OutputParser::new(
            ReasoningParser::new("<think>", "</think>", false),
            Some(ToolCallParser::new(
                ToolFormat::Hermes,
                vec!["get_weather".into()],
                false,
            )),
        );
        let events = run(
            &mut p,
            &[
                "<think>I could write <tool_call>{\"name\": \"get_weather\"}</tool_call></think>",
                "\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
            ],
        );
        assert_eq!(
            events[0],
            Event::ReasoningDelta(
                "I could write <tool_call>{\"name\": \"get_weather\"}</tool_call>".into()
            )
        );
        assert!(
            matches!(&events[1], Event::ToolCall(c) if c.arguments_json == r#"{"city":"Paris"}"#)
        );
        assert_eq!(events.len(), 2, "{events:?}");
        assert!(p.called());
    }

    #[test]
    fn a_plain_parser_passes_markup_through_as_content() {
        let mut p = OutputParser::plain();
        let events = run(&mut p, &["<tool_call>{}</tool_call>"]);
        assert_eq!(
            events,
            vec![Event::TextDelta("<tool_call>{}</tool_call>".into())]
        );
        assert!(!p.called());
    }
}
