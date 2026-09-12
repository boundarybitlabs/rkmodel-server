//! `POST /v1/responses`.
//!
//! The second adapter over `GenerateInput`. It differs from chat completions
//! only in the shapes it parses and emits, so neither endpoint holds logic the
//! other lacks.

use rkmodel_server_protocol::{FinishReason, GenerateInput, Message, Part, Role, ToolCall, Usage};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::chat::{Content, ContentPart};
use crate::error::ApiError;
use crate::tools::{self, FunctionDef};

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Input,
    /// Becomes a system message, placed first.
    pub instructions: Option<String>,

    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub reasoning: Option<ReasoningOptions>,

    #[serde(default)]
    pub stream: bool,

    /// Parsed by hand, so a malformed or built-in tool is a 400 naming the
    /// field.
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,

    // Refused, since each would change the shape or contract of the response.
    pub previous_response_id: Option<Value>,
    pub text: Option<Value>,
}

/// A function tool, flat: `{type, name, description, parameters, strict}`.
#[derive(Debug, Deserialize)]
struct ResponsesTool {
    #[serde(rename = "type")]
    kind: String,
    #[serde(flatten)]
    function: Option<FunctionDef>,
}

#[derive(Debug, Deserialize)]
pub struct ReasoningOptions {
    pub effort: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Input {
    /// One user message.
    Text(String),
    Items(Vec<InputItem>),
}

#[derive(Debug, Deserialize)]
pub struct InputItem {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<Content>,
    /// A reasoning item's summary, when it carries no full text.
    pub summary: Option<Vec<ContentPart>>,
    /// A `function_call` or `function_call_output` item's call.
    pub call_id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
    /// A `function_call_output` item's result, a string or input parts.
    pub output: Option<Value>,
}

fn present(value: &Option<Value>) -> bool {
    !matches!(value, None | Some(Value::Null))
}

impl ResponsesRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        if present(&self.previous_response_id) {
            return Err(ApiError::invalid_request(
                "Stored responses are not supported, so a previous response cannot be continued.",
                Some("previous_response_id"),
            ));
        }
        if let Some(text) = &self.text {
            let kind = text
                .get("format")
                .and_then(|f| f.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("text");
            if kind != "text" {
                return Err(ApiError::invalid_request(
                    format!("Only a text format of type text is supported, not {kind}."),
                    Some("text.format"),
                ));
            }
        }
        Ok(())
    }

    pub fn reasoning_flag(&self) -> Option<bool> {
        match self.reasoning.as_ref().and_then(|r| r.effort.as_deref()) {
            None => None,
            Some("none") | Some("minimal") => Some(false),
            Some(_) => Some(true),
        }
    }

    pub fn into_generate_input(self) -> Result<GenerateInput, ApiError> {
        let reasoning = self.reasoning_flag();
        let mut messages = Vec::new();

        if let Some(instructions) = &self.instructions {
            messages.push(Message::text(Role::System, instructions));
        }

        match &self.input {
            Input::Text(text) => messages.push(Message::text(Role::User, text)),
            Input::Items(items) => fold_items(items, &mut messages)?,
        }

        if messages.is_empty() {
            return Err(ApiError::invalid_request(
                "At least one input message is required.",
                Some("input"),
            ));
        }

        let mut tools = Vec::new();
        if let Some(value) = self.tools.filter(|v| !v.is_null()) {
            let declared: Vec<ResponsesTool> = serde_json::from_value(value).map_err(|e| {
                ApiError::invalid_request(format!("tools is malformed: {e}"), Some("tools"))
            })?;
            for tool in declared {
                tools::check_function_type(&tool.kind, "tools")?;
                let function = tool.function.ok_or_else(|| {
                    ApiError::invalid_request("A function tool needs a name.", Some("tools"))
                })?;
                tools.push(function.into_tool("tools")?);
            }
        }
        let tool_choice = self
            .tool_choice
            .as_ref()
            .filter(|v| !v.is_null())
            .map(|v| tools::parse_choice(v, "tool_choice"))
            .transpose()?;

        Ok(GenerateInput {
            messages,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_output_tokens,
            reasoning,
            tools,
            tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
        })
    }
}

/// Turns input items into messages.
///
/// The Responses API spreads one assistant turn over several items: a
/// reasoning item, perhaps a message, and a `function_call` item per call.
/// Templates want them as one message, so consecutive calls join the
/// assistant message before them, and a reasoning item becomes the reasoning
/// of the assistant turn that follows it.
fn fold_items(items: &[InputItem], messages: &mut Vec<Message>) -> Result<(), ApiError> {
    let mut reasoning: Option<String> = None;
    // Whether the last message is an assistant turn that a call may join.
    let mut joinable = false;

    for item in items {
        match item.kind.as_deref() {
            Some("reasoning") => {
                reasoning = Some(item.reasoning_text());
                joinable = false;
            }
            Some("function_call") => {
                let (Some(call_id), Some(name), Some(arguments)) =
                    (&item.call_id, &item.name, &item.arguments)
                else {
                    return Err(ApiError::invalid_request(
                        "A function_call item needs call_id, name and arguments.",
                        Some("input"),
                    ));
                };
                tools::check_arguments(arguments, "input")?;
                let call = ToolCall {
                    id: call_id.clone(),
                    name: name.clone(),
                    arguments_json: arguments.clone(),
                };
                match messages.last_mut() {
                    Some(last) if joinable => last.tool_calls.push(call),
                    _ => {
                        let mut message = Message::new(Role::Assistant, Vec::new());
                        message.reasoning = reasoning.take();
                        message.tool_calls.push(call);
                        messages.push(message);
                        joinable = true;
                    }
                }
            }
            Some("function_call_output") => {
                let Some(call_id) = &item.call_id else {
                    return Err(ApiError::invalid_request(
                        "A function_call_output item needs a call_id.",
                        Some("input"),
                    ));
                };
                let mut message = Message::text(Role::Tool, output_text(item.output.as_ref())?);
                message.tool_call_id = Some(call_id.clone());
                messages.push(message);
                reasoning = None;
                joinable = false;
            }
            None | Some("message") => {
                let mut message = item.to_message()?;
                joinable = message.role == Role::Assistant;
                if joinable {
                    message.reasoning = reasoning.take();
                } else {
                    // Reasoning before anything but the assistant's own turn
                    // answers nothing the template would render.
                    reasoning = None;
                }
                messages.push(message);
            }
            Some(other) => {
                return Err(ApiError::invalid_request(
                    format!("Unsupported input item type {other}."),
                    Some("input"),
                ))
            }
        }
    }
    Ok(())
}

/// A tool's result, which is a string or a list of input parts.
fn output_text(output: Option<&Value>) -> Result<String, ApiError> {
    match output {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match (
                    part.get("type").and_then(Value::as_str),
                    part.get("text").and_then(Value::as_str),
                ) {
                    (Some("input_text" | "output_text" | "text"), Some(t)) => text.push_str(t),
                    _ => {
                        return Err(ApiError::invalid_request(
                            "A function_call_output can only carry text.",
                            Some("input"),
                        ))
                    }
                }
            }
            Ok(text)
        }
        _ => Err(ApiError::invalid_request(
            "A function_call_output item needs an output.",
            Some("input"),
        )),
    }
}

impl InputItem {
    /// The full reasoning text, or the summary when that is all there is.
    fn reasoning_text(&self) -> String {
        let from = |parts: &[ContentPart]| -> String {
            parts.iter().filter_map(|p| p.text.as_deref()).collect()
        };
        match &self.content {
            Some(Content::Parts(parts)) if !parts.is_empty() => from(parts),
            Some(Content::Text(text)) => text.clone(),
            _ => self.summary.as_deref().map(from).unwrap_or_default(),
        }
    }

    fn to_message(&self) -> Result<Message, ApiError> {
        let role = match self.role.as_deref() {
            Some("system") | Some("developer") => Role::System,
            Some("user") | None => Role::User,
            Some("assistant") => Role::Assistant,
            Some(other) => {
                return Err(ApiError::invalid_request(
                    format!("Unsupported message role {other}."),
                    Some("input"),
                ))
            }
        };

        let parts = match &self.content {
            None => Vec::new(),
            Some(Content::Text(text)) => vec![Part::Text(text.clone())],
            Some(Content::Parts(parts)) => {
                let mut out = Vec::with_capacity(parts.len());
                for part in parts {
                    match (part.kind.as_str(), &part.text) {
                        ("input_text" | "output_text" | "text", Some(text)) => {
                            out.push(Part::Text(text.clone()))
                        }
                        ("input_image", _) => {
                            return Err(ApiError::invalid_request(
                                "Images are not supported yet.",
                                Some("input"),
                            ))
                        }
                        (kind, _) => {
                            return Err(ApiError::invalid_request(
                                format!("Unsupported content part {kind}."),
                                Some("input"),
                            ))
                        }
                    }
                }
                out
            }
        };

        Ok(Message::new(role, parts))
    }
}

// ---- response bodies -------------------------------------------------------

pub fn usage_json(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
    })
}

/// The reasoning item, which is present only when the model reasoned.
///
/// It carries the full text as `reasoning_text` rather than a `summary`, which
/// is how the Responses API represents open-weight reasoning.
pub fn reasoning_item(id: &str, text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": id,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    })
}

pub fn message_item(id: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "id": id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

/// A call as the Responses API writes it. The item's own id is not the call's
/// id: `call_id` is what a `function_call_output` answers.
pub fn function_call_item(call: &ToolCall, arguments: &str, status: &str) -> Value {
    json!({
        "type": "function_call",
        "id": function_call_item_id(call),
        "call_id": call.id,
        "name": call.name,
        "arguments": arguments,
        "status": status,
    })
}

fn function_call_item_id(call: &ToolCall) -> String {
    call.id.replacen("call_", "fc_", 1)
}

pub struct Ids {
    pub response: String,
    pub reasoning: String,
    pub message: String,
}

impl Ids {
    pub fn new() -> Ids {
        Ids {
            response: crate::id::generate("resp_"),
            reasoning: crate::id::generate("rs_"),
            message: crate::id::generate("msg_"),
        }
    }
}

/// The whole response object, which the streaming events also carry.
#[allow(clippy::too_many_arguments)]
pub fn response_json(
    ids: &Ids,
    created: u64,
    model: &str,
    status: &str,
    output: Vec<Value>,
    finish: Option<FinishReason>,
    usage: Option<&Usage>,
) -> Value {
    // A run cut off by its budget is incomplete rather than failed, and says so.
    let incomplete = match finish {
        Some(FinishReason::Length) => json!({"reason": "max_output_tokens"}),
        _ => Value::Null,
    };
    json!({
        "id": ids.response,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output,
        "incomplete_details": incomplete,
        "usage": usage.map(usage_json).unwrap_or(Value::Null),
    })
}

pub fn status_for(finish: FinishReason) -> &'static str {
    match finish {
        // A response that ends in calls is complete. The client answers them.
        FinishReason::Stop | FinishReason::ToolCalls => "completed",
        FinishReason::Length => "incomplete",
    }
}

/// The terminal event name matching a finish reason.
pub fn terminal_event(finish: FinishReason) -> &'static str {
    match finish {
        FinishReason::Stop | FinishReason::ToolCalls => "response.completed",
        FinishReason::Length => "response.incomplete",
    }
}

/// Assembles the finished body for `stream: false`.
#[allow(clippy::too_many_arguments)]
pub fn completed_json(
    ids: &Ids,
    created: u64,
    model: &str,
    text: &str,
    reasoning: Option<&str>,
    tool_calls: &[ToolCall],
    finish: FinishReason,
    usage: &Usage,
) -> Value {
    let mut output = Vec::new();
    if let Some(reasoning) = reasoning {
        output.push(reasoning_item(&ids.reasoning, reasoning));
    }
    // A turn that only calls tools has no message. One that says nothing at
    // all still gets an empty one, which is what a client reads for its text.
    if !text.is_empty() || tool_calls.is_empty() {
        output.push(message_item(&ids.message, text));
    }
    for call in tool_calls {
        output.push(function_call_item(call, &call.arguments_json, "completed"));
    }
    response_json(
        ids,
        created,
        model,
        status_for(finish),
        output,
        Some(finish),
        Some(usage),
    )
}

// ---- streaming -------------------------------------------------------------

/// A named event and its body, before the sequence number is added.
pub type NamedEvent = (&'static str, Value);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    Nothing,
    Reasoning,
    Message,
}

/// The output items of a streamed response, as the daemon's events arrive.
///
/// Items stream one at a time: whatever is open closes before the next opens,
/// and each takes the next output index. The finished items are the final
/// body's `output`, so the stream and a `stream: false` body agree.
pub struct StreamState {
    ids: Ids,
    open: Open,
    reasoning: String,
    text: String,
    /// The open message's id. A message after a call is a new item.
    message_id: String,
    messages: usize,
    calls: usize,
    output: Vec<Value>,
}

impl StreamState {
    pub fn new(ids: Ids) -> StreamState {
        let message_id = ids.message.clone();
        StreamState {
            ids,
            open: Open::Nothing,
            reasoning: String::new(),
            text: String::new(),
            message_id,
            messages: 0,
            calls: 0,
            output: Vec::new(),
        }
    }

    pub fn ids(&self) -> &Ids {
        &self.ids
    }

    fn index(&self) -> usize {
        self.output.len()
    }

    pub fn reasoning_delta(&mut self, delta: &str) -> Vec<NamedEvent> {
        let mut events = Vec::new();
        if self.open != Open::Reasoning {
            events.extend(self.close());
            self.open = Open::Reasoning;
            self.reasoning.clear();
            events.push((
                "response.output_item.added",
                json!({
                    "output_index": self.index(),
                    "item": reasoning_item(&self.ids.reasoning, ""),
                }),
            ));
        }
        self.reasoning.push_str(delta);
        events.push((
            "response.reasoning_text.delta",
            json!({
                "item_id": self.ids.reasoning,
                "output_index": self.index(),
                "content_index": 0,
                "delta": delta,
            }),
        ));
        events
    }

    pub fn text_delta(&mut self, delta: &str) -> Vec<NamedEvent> {
        let mut events = Vec::new();
        if self.open != Open::Message {
            events.extend(self.close());
            events.extend(self.open_message());
        }
        self.text.push_str(delta);
        events.push((
            "response.output_text.delta",
            json!({
                "item_id": self.message_id,
                "output_index": self.index(),
                "content_index": 0,
                "delta": delta,
            }),
        ));
        events
    }

    /// A call arrives whole, so its item opens, carries every argument in one
    /// delta, and closes.
    pub fn tool_call(&mut self, call: &ToolCall) -> Vec<NamedEvent> {
        let mut events = self.close();
        let index = self.index();
        let item_id = function_call_item_id(call);
        events.push((
            "response.output_item.added",
            json!({
                "output_index": index,
                "item": function_call_item(call, "", "in_progress"),
            }),
        ));
        events.push((
            "response.function_call_arguments.delta",
            json!({"item_id": item_id, "output_index": index, "delta": call.arguments_json}),
        ));
        events.push((
            "response.function_call_arguments.done",
            json!({
                "item_id": item_id,
                "output_index": index,
                "name": call.name,
                "arguments": call.arguments_json,
            }),
        ));
        let item = function_call_item(call, &call.arguments_json, "completed");
        events.push((
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        ));
        self.output.push(item);
        self.calls += 1;
        events
    }

    /// Closes what is open. A run that produced neither a message nor a call,
    /// such as one cut off while reasoning, still gets an empty message, so
    /// the stream matches the `stream: false` body.
    pub fn finish(&mut self) -> Vec<NamedEvent> {
        let mut events = self.close();
        if self.messages == 0 && self.calls == 0 {
            events.extend(self.open_message());
            events.extend(self.close());
        }
        events
    }

    /// The finished items, for the terminal event's response.
    pub fn output(&self) -> Vec<Value> {
        self.output.clone()
    }

    fn open_message(&mut self) -> Vec<NamedEvent> {
        if self.messages > 0 {
            self.message_id = crate::id::generate("msg_");
        }
        self.messages += 1;
        self.open = Open::Message;
        self.text.clear();
        vec![
            (
                "response.output_item.added",
                json!({
                    "output_index": self.index(),
                    "item": message_item(&self.message_id, ""),
                }),
            ),
            (
                "response.content_part.added",
                json!({
                    "item_id": self.message_id,
                    "output_index": self.index(),
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }),
            ),
        ]
    }

    fn close(&mut self) -> Vec<NamedEvent> {
        let index = self.index();
        let events = match self.open {
            Open::Nothing => return Vec::new(),
            Open::Reasoning => {
                let item = reasoning_item(&self.ids.reasoning, &self.reasoning);
                let events = vec![
                    (
                        "response.reasoning_text.done",
                        json!({
                            "item_id": self.ids.reasoning,
                            "output_index": index,
                            "content_index": 0,
                            "text": self.reasoning,
                        }),
                    ),
                    (
                        "response.output_item.done",
                        json!({"output_index": index, "item": item.clone()}),
                    ),
                ];
                self.output.push(item);
                events
            }
            Open::Message => {
                let item = message_item(&self.message_id, &self.text);
                let events = vec![
                    (
                        "response.output_text.done",
                        json!({
                            "item_id": self.message_id,
                            "output_index": index,
                            "content_index": 0,
                            "text": self.text,
                        }),
                    ),
                    (
                        "response.content_part.done",
                        json!({
                            "item_id": self.message_id,
                            "output_index": index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": self.text, "annotations": []},
                        }),
                    ),
                    (
                        "response.output_item.done",
                        json!({"output_index": index, "item": item.clone()}),
                    ),
                ];
                self.output.push(item);
                events
            }
        };
        self.open = Open::Nothing;
        events
    }
}
