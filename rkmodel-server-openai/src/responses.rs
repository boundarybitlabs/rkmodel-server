//! `POST /v1/responses`.
//!
//! The second adapter over `GenerateInput`. It differs from chat completions
//! only in the shapes it parses and emits, so neither endpoint holds logic the
//! other lacks.

use rkmodel_server_protocol::{FinishReason, GenerateInput, Message, Part, Role, Usage};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::chat::Content;
use crate::error::ApiError;

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

    // Refused, since each would change the shape or contract of the response.
    pub previous_response_id: Option<Value>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub text: Option<Value>,
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
        if present(&self.tools) {
            return Err(ApiError::invalid_request(
                "Tools are not supported.",
                Some("tools"),
            ));
        }
        if present(&self.tool_choice) {
            return Err(ApiError::invalid_request(
                "Tool choice is not supported.",
                Some("tool_choice"),
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
            Input::Items(items) => {
                for item in items {
                    if let Some(message) = item.to_message()? {
                        messages.push(message);
                    }
                }
            }
        }

        if messages.is_empty() {
            return Err(ApiError::invalid_request(
                "At least one input message is required.",
                Some("input"),
            ));
        }

        Ok(GenerateInput {
            messages,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_output_tokens,
            reasoning,
            ..Default::default()
        })
    }
}

impl InputItem {
    /// `None` for an item that carries no turn, such as a reasoning item echoed
    /// back from an earlier response.
    fn to_message(&self) -> Result<Option<Message>, ApiError> {
        // The daemon ignores reasoning on input, and the model's own template
        // drops it from earlier turns, so echoed reasoning is dropped here too.
        if self.kind.as_deref() == Some("reasoning") {
            return Ok(None);
        }
        match self.kind.as_deref() {
            None | Some("message") => {}
            Some(other) => {
                return Err(ApiError::invalid_request(
                    format!("Unsupported input item type {other}."),
                    Some("input"),
                ))
            }
        }

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

        Ok(Some(Message::new(role, parts)))
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
pub fn completed_json(
    ids: &Ids,
    created: u64,
    model: &str,
    text: &str,
    reasoning: Option<&str>,
    finish: FinishReason,
    usage: &Usage,
) -> Value {
    let mut output = Vec::new();
    if let Some(reasoning) = reasoning {
        output.push(reasoning_item(&ids.reasoning, reasoning));
    }
    output.push(message_item(&ids.message, text));
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
