//! `POST /v1/chat/completions`.
//!
//! A thin adapter over `GenerateInput`. `/v1/responses` builds the same thing,
//! so neither endpoint holds logic the other lacks.

use rkmodel_server_protocol::{FinishReason, GenerateInput, Message, Part, Role, Usage};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::ApiError;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub reasoning_effort: Option<String>,
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,

    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,

    // Fields that change the shape or contract of the response. Present means
    // refused, with the field named.
    pub n: Option<u32>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub logprobs: Option<Value>,
    pub response_format: Option<Value>,
    // Everything unlisted is ignored, which is what OpenAI-compatible servers
    // generally do and what keeps new SDK versions working.
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// Clients set up for vLLM send reasoning this way.
#[derive(Debug, Deserialize)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

impl ChatRequest {
    /// Refuses what would otherwise change the response's shape, and names the
    /// field so a client can see what to drop.
    pub fn validate(&self) -> Result<(), ApiError> {
        if let Some(n) = self.n {
            if n > 1 {
                return Err(ApiError::invalid_request(
                    "Only one choice per request is supported.",
                    Some("n"),
                ));
            }
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
        if self.logprobs.as_ref().is_some_and(|v| v == &json!(true)) {
            return Err(ApiError::invalid_request(
                "Log probabilities are not supported.",
                Some("logprobs"),
            ));
        }
        if let Some(format) = &self.response_format {
            let kind = format.get("type").and_then(Value::as_str).unwrap_or("text");
            if kind != "text" {
                return Err(ApiError::invalid_request(
                    format!("Only a response_format of type text is supported, not {kind}."),
                    Some("response_format"),
                ));
            }
        }
        if self.messages.is_empty() {
            return Err(ApiError::invalid_request(
                "At least one message is required.",
                Some("messages"),
            ));
        }
        Ok(())
    }

    /// `max_completion_tokens` wins when both are sent.
    pub fn budget(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// How this request asks for reasoning. `None` takes the model's default.
    ///
    /// `chat_template_kwargs` is the more explicit of the two, so it wins.
    pub fn reasoning(&self) -> Option<bool> {
        if let Some(kwargs) = &self.chat_template_kwargs {
            if let Some(on) = kwargs.enable_thinking {
                return Some(on);
            }
        }
        match self.reasoning_effort.as_deref() {
            None => None,
            Some("none") | Some("minimal") => Some(false),
            // Qwen3 has no levels, so low, medium and high are the same.
            Some(_) => Some(true),
        }
    }

    pub fn into_generate_input(self) -> Result<GenerateInput, ApiError> {
        let budget = self.budget();
        let reasoning = self.reasoning();
        let mut messages = Vec::with_capacity(self.messages.len());
        for m in &self.messages {
            messages.push(m.to_message()?);
        }
        Ok(GenerateInput {
            messages,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: budget,
            reasoning,
            ..Default::default()
        })
    }
}

fn present(value: &Option<Value>) -> bool {
    !matches!(value, None | Some(Value::Null))
}

impl ChatMessage {
    fn to_message(&self) -> Result<Message, ApiError> {
        let role = match self.role.as_str() {
            "system" => Role::System,
            // The Responses API renamed system to developer. Same thing here.
            "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => {
                return Err(ApiError::invalid_request(
                    format!("Unsupported message role {other}."),
                    Some("messages"),
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
                        ("text" | "input_text" | "output_text", Some(text)) => {
                            out.push(Part::Text(text.clone()))
                        }
                        (kind, _) => {
                            return Err(ApiError::invalid_request(
                                format!("Unsupported content part {kind}."),
                                Some("messages"),
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

pub fn finish_str(finish: FinishReason) -> &'static str {
    match finish {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    }
}

pub fn usage_json(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
    })
}

/// The whole answer, for `stream: false`.
pub fn completion_json(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    reasoning: Option<String>,
    finish: FinishReason,
    usage: &Usage,
) -> Value {
    let mut message = json!({"role": "assistant", "content": text});
    // The name DeepSeek's API, vLLM and llama.cpp's server use, so clients that
    // already read reasoning from a chat completion find it there. Absent when
    // there was no reasoning.
    if let Some(reasoning) = reasoning {
        message["reasoning_content"] = json!(reasoning);
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_str(finish),
        }],
        "usage": usage_json(usage),
    })
}

/// One `chat.completion.chunk`, with whatever delta it carries.
pub fn chunk_json(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish: Option<&str>,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
        }],
    })
}

/// The final chunk when `stream_options.include_usage` is set, which carries no
/// choice of its own.
pub fn usage_chunk_json(id: &str, created: u64, model: &str, usage: &Usage) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage_json(usage),
    })
}
