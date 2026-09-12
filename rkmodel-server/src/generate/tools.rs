//! Tool calls in a model's output: each format's markers, the start of a call
//! the daemon writes to force one, and parsing calls back out of the stream.
//!
//! The model's template renders the declarations and the earlier calls. This
//! only reads what the model writes next.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rkmodel_server_protocol::{ToolCall, ToolChoice};
use serde_json::{Map, Value};

use crate::config::ToolFormat;
use crate::generate::reasoning::partial_suffix;

/// Gemma's string delimiter. A special token, so it cannot occur inside a
/// string, and its arguments need no escaping rule.
const GEMMA_QUOTE: &str = "<|\"|>";

impl ToolFormat {
    /// The markers around one call.
    fn open(self) -> &'static str {
        match self {
            ToolFormat::Hermes => "<tool_call>",
            ToolFormat::Gemma4 => "<|tool_call>",
        }
    }

    fn close(self) -> &'static str {
        match self {
            ToolFormat::Hermes => "</tool_call>",
            ToolFormat::Gemma4 => "<tool_call|>",
        }
    }

    /// Written by a model that expects the result in the same turn. Gemma's
    /// template continues its turn with the result, so after a call it may
    /// write this and go on to invent one. The run stops there instead.
    fn stop(self) -> Option<&'static str> {
        match self {
            ToolFormat::Hermes => None,
            ToolFormat::Gemma4 => Some("<|tool_response>"),
        }
    }

    /// Strings stripped from user text and tool results, so neither can forge
    /// a call or a result. Gemma's are special tokens and would be stripped
    /// anyway; Qwen's are not.
    pub fn reserved(self) -> &'static [&'static str] {
        match self {
            ToolFormat::Hermes => &[
                "<tool_call>",
                "</tool_call>",
                "<tool_response>",
                "</tool_response>",
            ],
            ToolFormat::Gemma4 => &[
                "<|tool_call>",
                "<tool_call|>",
                "<|tool_response>",
                "<tool_response|>",
                "<|tool>",
                "<tool|>",
                GEMMA_QUOTE,
            ],
        }
    }

    /// The start of a call, for a choice that forces one: what is appended to
    /// the prompt, and what of it follows the open marker, which the parser is
    /// primed with. `None` for a choice that forces nothing.
    ///
    /// RKLLM cannot constrain decoding, but the prompt is the daemon's to
    /// write, so the model is handed a call already begun.
    pub fn forced_start(self, choice: &ToolChoice) -> Option<ForcedStart> {
        let body = match (self, choice) {
            (_, ToolChoice::None | ToolChoice::Auto) => return None,
            (ToolFormat::Hermes, ToolChoice::Required) => "\n{\"name\": \"".to_string(),
            (ToolFormat::Hermes, ToolChoice::Function(name)) => {
                format!("\n{{\"name\": \"{name}\", \"arguments\": ")
            }
            (ToolFormat::Gemma4, ToolChoice::Required) => "call:".to_string(),
            (ToolFormat::Gemma4, ToolChoice::Function(name)) => format!("call:{name}{{"),
        };
        Some(ForcedStart {
            prompt: format!("{}{body}", self.open()),
            body,
        })
    }

    /// One call's body, between its markers.
    fn parse(self, body: &str) -> Option<(String, Value)> {
        match self {
            ToolFormat::Hermes => parse_hermes(body),
            ToolFormat::Gemma4 => parse_gemma(body),
        }
    }
}

/// See [`ToolFormat::forced_start`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForcedStart {
    /// Appended to the rendered prompt.
    pub prompt: String,
    /// The part of `prompt` after the open marker.
    pub body: String,
}

/// `{"name": "get_weather", "arguments": {"city": "Paris"}}`.
///
/// Arguments given as a JSON string are unwrapped, and `parameters` is accepted
/// for `arguments`, since Qwen models write both.
fn parse_hermes(body: &str) -> Option<(String, Value)> {
    let Value::Object(mut call) = serde_json::from_str(body.trim()).ok()? else {
        return None;
    };
    let Value::String(name) = call.remove("name")? else {
        return None;
    };
    let arguments = match call
        .remove("arguments")
        .or_else(|| call.remove("parameters"))
    {
        None => Value::Object(Map::new()),
        Some(Value::String(s)) => serde_json::from_str(&s).ok()?,
        Some(v) => v,
    };
    arguments.is_object().then_some((name, arguments))
}

/// `call:get_weather{city:<|"|>Paris<|"|>,unit:<|"|>c<|"|>}`.
fn parse_gemma(body: &str) -> Option<(String, Value)> {
    let rest = body.trim().strip_prefix("call:")?;
    let brace = rest.find('{')?;
    let name = &rest[..brace];
    if name.is_empty() || !name.chars().all(is_name_char) {
        return None;
    }
    let mut parser = GemmaValue {
        text: &rest[brace..],
    };
    let arguments = parser.object()?;
    parser
        .text
        .trim()
        .is_empty()
        .then(|| (name.to_string(), arguments))
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.'
}

/// Gemma's argument syntax: JSON with bare keys, and strings between `<|"|>`
/// rather than quotes. Keys written as strings are accepted too, since that is
/// how its template writes declarations.
struct GemmaValue<'a> {
    text: &'a str,
}

impl GemmaValue<'_> {
    fn skip_space(&mut self) {
        self.text = self.text.trim_start();
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_space();
        match self.text.strip_prefix(token) {
            Some(rest) => {
                self.text = rest;
                true
            }
            None => false,
        }
    }

    fn value(&mut self) -> Option<Value> {
        self.skip_space();
        if self.text.starts_with('{') {
            self.object()
        } else if self.text.starts_with('[') {
            self.array()
        } else if self.text.starts_with(GEMMA_QUOTE) {
            self.string().map(Value::String)
        } else {
            self.literal()
        }
    }

    fn object(&mut self) -> Option<Value> {
        if !self.eat("{") {
            return None;
        }
        let mut map = Map::new();
        if self.eat("}") {
            return Some(Value::Object(map));
        }
        loop {
            let key = self.key()?;
            if !self.eat(":") {
                return None;
            }
            let value = self.value()?;
            map.insert(key, value);
            if self.eat("}") {
                return Some(Value::Object(map));
            }
            if !self.eat(",") {
                return None;
            }
        }
    }

    fn array(&mut self) -> Option<Value> {
        if !self.eat("[") {
            return None;
        }
        let mut items = Vec::new();
        if self.eat("]") {
            return Some(Value::Array(items));
        }
        loop {
            items.push(self.value()?);
            if self.eat("]") {
                return Some(Value::Array(items));
            }
            if !self.eat(",") {
                return None;
            }
        }
    }

    fn key(&mut self) -> Option<String> {
        self.skip_space();
        if self.text.starts_with(GEMMA_QUOTE) {
            return self.string();
        }
        let end = self.text.find(':')?;
        let key = self.text[..end].trim();
        if key.is_empty() || !key.chars().all(is_name_char) {
            return None;
        }
        self.text = &self.text[end..];
        Some(key.to_string())
    }

    fn string(&mut self) -> Option<String> {
        let rest = self.text.strip_prefix(GEMMA_QUOTE)?;
        let end = rest.find(GEMMA_QUOTE)?;
        self.text = &rest[end + GEMMA_QUOTE.len()..];
        Some(rest[..end].to_string())
    }

    /// A number, `true`, `false` or `null`. Anything else is not a value, and
    /// the call does not parse.
    fn literal(&mut self) -> Option<Value> {
        let end = self.text.find([',', '}', ']']).unwrap_or(self.text.len());
        let word = self.text[..end].trim();
        let value = match word {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" => Value::Null,
            _ => Value::Number(serde_json::from_str(word).ok()?),
        };
        self.text = &self.text[end..];
        Some(value)
    }
}

/// A fresh call id. Nothing stores these; they only have to be unique among
/// the calls one conversation carries, which a clock beside a counter is.
fn call_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{nanos:x}{n:x}")
}

/// One piece of parsed output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolPiece {
    Text(String),
    Call(ToolCall),
}

/// Pulls calls out of a model's text as it streams.
pub struct ToolCallParser {
    format: ToolFormat,
    declared: Vec<String>,
    stop_after_first: bool,
    inside: bool,
    /// Outside a call, a tail that could still become a marker, or text that
    /// is only whitespace. Inside one, the body so far.
    held: String,
    calls: u32,
    stopped: bool,
}

impl ToolCallParser {
    /// `declared` is the tools the request offered. A call naming anything
    /// else is left as text.
    pub fn new(format: ToolFormat, declared: Vec<String>, stop_after_first: bool) -> Self {
        ToolCallParser {
            format,
            declared,
            stop_after_first,
            inside: false,
            held: String::new(),
            calls: 0,
            stopped: false,
        }
    }

    /// For a run whose prompt already opened a call, starting inside it with
    /// what the prompt wrote after the open marker.
    pub fn primed(mut self, body: &str) -> Self {
        self.inside = true;
        self.held = body.to_string();
        self
    }

    /// How many calls have been emitted.
    pub fn calls(&self) -> u32 {
        self.calls
    }

    /// Feeds one chunk of text. The flag is true when the run should stop.
    pub fn push(&mut self, chunk: &str) -> (Vec<ToolPiece>, bool) {
        let mut out = Vec::new();
        if self.stopped {
            return (out, true);
        }
        self.held.push_str(chunk);

        loop {
            if self.inside {
                let Some(at) = self.held.find(self.format.close()) else {
                    return (out, false);
                };
                let body = self.held[..at].to_string();
                self.held.drain(..at + self.format.close().len());
                self.inside = false;

                match self.format.parse(&body) {
                    Some((name, arguments)) if self.declared.contains(&name) => {
                        self.calls += 1;
                        out.push(ToolPiece::Call(ToolCall {
                            id: call_id(),
                            name,
                            arguments_json: arguments.to_string(),
                        }));
                        if self.stop_after_first {
                            self.stopped = true;
                            return (out, true);
                        }
                    }
                    // Not a call anyone can act on, so the client sees what the
                    // model wrote.
                    _ => out.push(ToolPiece::Text(format!(
                        "{}{body}{}",
                        self.format.open(),
                        self.format.close()
                    ))),
                }
                continue;
            }

            let open = self.held.find(self.format.open());
            let stop = self.format.stop().and_then(|s| self.held.find(s));
            match (open, stop) {
                (_, Some(s)) if open.is_none_or(|o| s < o) => {
                    let before = self.held[..s].to_string();
                    self.held.clear();
                    self.emit_text(&mut out, &before, false);
                    self.stopped = true;
                    return (out, true);
                }
                (Some(o), _) => {
                    let before = self.held[..o].to_string();
                    self.held.drain(..o + self.format.open().len());
                    // Whitespace before a call is only layout.
                    self.emit_text(&mut out, &before, true);
                    self.inside = true;
                }
                _ => {
                    let keep = [Some(self.format.open()), self.format.stop()]
                        .into_iter()
                        .flatten()
                        .map(|m| partial_suffix(&self.held, m))
                        .max()
                        .unwrap_or(0);
                    let split = self.held.len() - keep;
                    let ready = self.held[..split].to_string();
                    // Whitespace alone waits: it is layout if a call follows,
                    // and content if text does.
                    if ready.trim().is_empty() {
                        return (out, false);
                    }
                    self.held.drain(..split);
                    self.emit_text(&mut out, &ready, false);
                    return (out, false);
                }
            }
        }
    }

    fn emit_text(&self, out: &mut Vec<ToolPiece>, text: &str, before_call: bool) {
        if text.is_empty() || (before_call && text.trim().is_empty()) {
            return;
        }
        out.push(ToolPiece::Text(text.to_string()));
    }

    /// Flushes what is held. A call cut off before its close marker was never
    /// a call, and trailing whitespace after calls is only layout.
    pub fn finish(&mut self) -> Vec<ToolPiece> {
        let held = std::mem::take(&mut self.held);
        if self.stopped {
            return Vec::new();
        }
        if self.inside {
            self.inside = false;
            return vec![ToolPiece::Text(format!("{}{held}", self.format.open()))];
        }
        if held.is_empty() || (self.calls > 0 && held.trim().is_empty()) {
            return Vec::new();
        }
        vec![ToolPiece::Text(held)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hermes() -> ToolCallParser {
        ToolCallParser::new(ToolFormat::Hermes, vec!["get_weather".into()], false)
    }

    fn gemma() -> ToolCallParser {
        ToolCallParser::new(ToolFormat::Gemma4, vec!["get_weather".into()], false)
    }

    /// Feeds every chunk, returning the pieces with call ids blanked and
    /// whether the run was told to stop.
    fn run(p: &mut ToolCallParser, chunks: &[&str]) -> (Vec<ToolPiece>, bool) {
        let mut out = Vec::new();
        let mut stopped = false;
        for c in chunks {
            let (pieces, stop) = p.push(c);
            out.extend(pieces);
            if stop {
                stopped = true;
                break;
            }
        }
        out.extend(p.finish());
        for piece in &mut out {
            if let ToolPiece::Call(call) = piece {
                assert!(call.id.starts_with("call_"), "{call:?}");
                call.id.clear();
            }
        }
        (out, stopped)
    }

    fn text(s: &str) -> ToolPiece {
        ToolPiece::Text(s.into())
    }

    fn call(name: &str, arguments: &str) -> ToolPiece {
        ToolPiece::Call(ToolCall {
            id: String::new(),
            name: name.into(),
            arguments_json: arguments.into(),
        })
    }

    /// Splits a string into one-character chunks, the worst case for markers.
    fn chars(s: &str) -> Vec<String> {
        s.chars().map(String::from).collect()
    }

    #[test]
    fn a_hermes_call_as_qwen3_writes_it_on_the_board() {
        // The callbacks Qwen3-0.6B produced, token by token, with the end of
        // turn already dropped by the worker.
        let chunks = [
            "<tool_call>",
            "\n",
            "{\"",
            "name",
            "\":",
            " \"",
            "get",
            "_weather",
            "\",",
            " \"",
            "arguments",
            "\":",
            " {\"",
            "city",
            "\":",
            " \"",
            "Paris",
            "\"}}\n",
            "</tool_call>",
        ];
        let (out, stopped) = run(&mut hermes(), &chunks);
        assert_eq!(out, vec![call("get_weather", r#"{"city":"Paris"}"#)]);
        assert!(!stopped);
    }

    #[test]
    fn a_gemma_call_as_gemma_writes_it_on_the_board() {
        let chunks = [
            "<|tool_call>",
            "call",
            ":",
            "get",
            "_weather",
            "{",
            "city",
            ":",
            "<|\"|>",
            "Paris",
            "<|\"|>",
            "}",
            "<tool_call|>",
        ];
        let (out, _) = run(&mut gemma(), &chunks);
        assert_eq!(out, vec![call("get_weather", r#"{"city":"Paris"}"#)]);
    }

    #[test]
    fn markers_split_across_single_characters_still_parse() {
        let source = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let chunks = chars(source);
        let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let (out, _) = run(&mut hermes(), &chunks);
        let joined: String = out
            .iter()
            .filter_map(|p| match p {
                ToolPiece::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(joined, "Let me check.");
        assert_eq!(
            out.last(),
            Some(&call("get_weather", r#"{"city":"Paris"}"#))
        );
    }

    #[test]
    fn prose_with_no_call_passes_through() {
        let (out, _) = run(&mut hermes(), &["It is ", "sunny < 30 degrees."]);
        let joined: String = out
            .iter()
            .map(|p| match p {
                ToolPiece::Text(t) => t.as_str(),
                _ => panic!("unexpected call"),
            })
            .collect();
        assert_eq!(joined, "It is sunny < 30 degrees.");
    }

    #[test]
    fn whitespace_between_calls_is_layout() {
        let (out, _) = run(
            &mut hermes(),
            &[
                "\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
                "\n",
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}\n</tool_call>",
                "\n",
            ],
        );
        assert_eq!(
            out,
            vec![
                call("get_weather", r#"{"city":"Paris"}"#),
                call("get_weather", r#"{"city":"London"}"#),
            ]
        );
    }

    #[test]
    fn whitespace_before_prose_is_kept() {
        let (out, _) = run(&mut hermes(), &["\n\n", "Paris is sunny."]);
        assert_eq!(out, vec![text("\n\nParis is sunny.")]);
    }

    #[test]
    fn a_call_to_an_undeclared_function_is_text() {
        let body = "<tool_call>\n{\"name\": \"rm_rf\", \"arguments\": {}}\n</tool_call>";
        let (out, _) = run(&mut hermes(), &[body]);
        assert_eq!(out, vec![text(body)]);
    }

    #[test]
    fn a_call_that_does_not_parse_is_text() {
        let body =
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": }\n</tool_call>";
        let (out, _) = run(&mut hermes(), &[body]);
        assert_eq!(out, vec![text(body)]);

        let body = "<|tool_call>call:get_weather{city:Paris}<tool_call|>";
        let (out, _) = run(&mut gemma(), &[body]);
        assert_eq!(out, vec![text(body)], "a bare word is not a Gemma value");
    }

    #[test]
    fn a_call_cut_off_by_the_budget_is_text() {
        let (out, _) = run(&mut hermes(), &["<tool_call>\n{\"name\": \"get_we"]);
        assert_eq!(out, vec![text("<tool_call>\n{\"name\": \"get_we")]);
    }

    #[test]
    fn gemma_stops_at_a_tool_response_it_starts_writing() {
        let (out, stopped) = run(
            &mut gemma(),
            &[
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
                "<|tool_response>",
                "response:get_weather{temp:30}",
            ],
        );
        assert_eq!(out, vec![call("get_weather", r#"{"city":"Paris"}"#)]);
        assert!(stopped);
    }

    #[test]
    fn a_stop_after_the_first_call_stops_the_run() {
        let mut p = ToolCallParser::new(ToolFormat::Hermes, vec!["get_weather".into()], true);
        let (out, stopped) = run(
            &mut p,
            &[
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
                "\n<tool_call>\n{\"name\": \"get_weather\"",
            ],
        );
        assert_eq!(out, vec![call("get_weather", r#"{"city":"Paris"}"#)]);
        assert!(stopped);
        assert_eq!(p.calls(), 1);
    }

    #[test]
    fn hermes_accepts_string_arguments_and_parameters() {
        assert_eq!(
            parse_hermes(r#"{"name": "f", "arguments": "{\"a\": 1}"}"#),
            Some(("f".into(), serde_json::json!({"a": 1})))
        );
        assert_eq!(
            parse_hermes(r#"{"name": "f", "parameters": {"a": 1}}"#),
            Some(("f".into(), serde_json::json!({"a": 1})))
        );
        assert_eq!(
            parse_hermes(r#"{"name": "f"}"#),
            Some(("f".into(), serde_json::json!({})))
        );
        assert_eq!(parse_hermes(r#"{"name": "f", "arguments": [1]}"#), None);
    }

    #[test]
    fn gemma_arguments_cover_every_value_kind() {
        let body = r#"call:plan{when:<|"|>2026-09-12, 9am<|"|>,count:3,ratio:-0.5,ok:true,
            nothing:null,tags:[<|"|>a<|"|>,<|"|>b<|"|>],where:{<|"|>lat<|"|>:44.9,lon:-93.2},empty:{}}"#;
        let (name, arguments) = parse_gemma(body).unwrap();
        assert_eq!(name, "plan");
        assert_eq!(
            arguments.to_string(),
            r#"{"when":"2026-09-12, 9am","count":3,"ratio":-0.5,"ok":true,"nothing":null,"tags":["a","b"],"where":{"lat":44.9,"lon":-93.2},"empty":{}}"#
        );
        assert_eq!(
            parse_gemma("call:f{}"),
            Some(("f".into(), serde_json::json!({})))
        );
        assert_eq!(parse_gemma("call:f{a:1} trailing"), None);
        assert_eq!(parse_gemma("call:{a:1}"), None);
    }

    #[test]
    fn forcing_a_function_primes_the_parser_with_what_the_prompt_wrote() {
        let choice = ToolChoice::Function("get_weather".into());

        let start = ToolFormat::Gemma4.forced_start(&choice).unwrap();
        assert_eq!(start.prompt, "<|tool_call>call:get_weather{");
        // What Gemma 4 E2B wrote on the board after that prompt.
        let mut p = gemma().primed(&start.body);
        let (out, _) = run(
            &mut p,
            &[
                "city",
                ":",
                "<|\"|>",
                "Paris",
                "<|\"|>",
                "}",
                "<tool_call|>",
            ],
        );
        assert_eq!(out, vec![call("get_weather", r#"{"city":"Paris"}"#)]);

        let start = ToolFormat::Hermes.forced_start(&choice).unwrap();
        assert_eq!(
            start.prompt,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        );
        let mut p = hermes().primed(&start.body);
        let (out, _) = run(&mut p, &["{\"city\": \"Paris\"}}\n</tool_call>"]);
        assert_eq!(out, vec![call("get_weather", r#"{"city":"Paris"}"#)]);
    }

    #[test]
    fn required_starts_a_call_without_naming_one() {
        let start = ToolFormat::Hermes
            .forced_start(&ToolChoice::Required)
            .unwrap();
        assert_eq!(start.prompt, "<tool_call>\n{\"name\": \"");
        let mut p = hermes().primed(&start.body);
        let (out, _) = run(&mut p, &["get_weather\", \"arguments\": {}}\n</tool_call>"]);
        assert_eq!(out, vec![call("get_weather", "{}")]);

        assert_eq!(
            ToolFormat::Gemma4
                .forced_start(&ToolChoice::Required)
                .unwrap()
                .prompt,
            "<|tool_call>call:"
        );
        assert_eq!(ToolFormat::Gemma4.forced_start(&ToolChoice::Auto), None);
        assert_eq!(ToolFormat::Hermes.forced_start(&ToolChoice::None), None);
    }
}
