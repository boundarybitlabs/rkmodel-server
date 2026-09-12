//! Tools, in both APIs' shapes.
//!
//! Chat completions nests a function under `function`, and the Responses API
//! flattens it. Everything past parsing is the same, so it lives here.

use rkmodel_server_protocol::{Tool, ToolCall, ToolChoice};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::ApiError;

/// A function's declaration, as both APIs spell its fields.
#[derive(Debug, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    /// Asks for arguments guaranteed to match the schema, which nothing here
    /// can enforce. Accepted anyway: the official SDK's
    /// `pydantic_function_tool` sets it on every tool, and refusing it would
    /// refuse most agent frameworks.
    #[allow(dead_code)]
    pub strict: Option<bool>,
}

impl FunctionDef {
    pub fn into_tool(self, param: &'static str) -> Result<Tool, ApiError> {
        let parameters_json = match self.parameters {
            None | Some(Value::Null) => None,
            Some(schema @ Value::Object(_)) => Some(schema.to_string()),
            Some(_) => {
                return Err(ApiError::invalid_request(
                    format!(
                        "The parameters of tool {} must be a JSON Schema object.",
                        self.name
                    ),
                    Some(param),
                ))
            }
        };
        Ok(Tool {
            name: self.name,
            description: self.description,
            parameters_json,
        })
    }
}

/// Only function tools can be served. A built-in tool, such as `web_search`,
/// would need the server to act on the model's behalf.
pub fn check_function_type(kind: &str, param: &'static str) -> Result<(), ApiError> {
    if kind == "function" {
        Ok(())
    } else {
        Err(ApiError::invalid_request(
            format!("Only function tools are supported, not {kind}."),
            Some(param),
        ))
    }
}

/// `"none"`, `"auto"`, `"required"`, or an object naming a function: under
/// `function.name` for chat completions, or `name` for the Responses API.
pub fn parse_choice(value: &Value, param: &'static str) -> Result<ToolChoice, ApiError> {
    match value {
        Value::String(s) => match s.as_str() {
            "none" => Ok(ToolChoice::None),
            "auto" => Ok(ToolChoice::Auto),
            "required" => Ok(ToolChoice::Required),
            other => Err(ApiError::invalid_request(
                format!("Unknown tool_choice {other}."),
                Some(param),
            )),
        },
        Value::Object(choice) => {
            let kind = choice.get("type").and_then(Value::as_str).unwrap_or("");
            check_function_type(kind, param)?;
            let name = choice
                .get("name")
                .or_else(|| choice.get("function").and_then(|f| f.get("name")))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ApiError::invalid_request(
                        "A function tool_choice must name the function.",
                        Some(param),
                    )
                })?;
            Ok(ToolChoice::Function(name.to_string()))
        }
        _ => Err(ApiError::invalid_request(
            "tool_choice must be a string or an object.",
            Some(param),
        )),
    }
}

/// An earlier call's arguments, which a template reads as an object.
pub fn check_arguments(arguments: &str, param: &'static str) -> Result<(), ApiError> {
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(_)) => Ok(()),
        _ => Err(ApiError::invalid_request(
            "A tool call's arguments must be a JSON object, encoded as a string.",
            Some(param),
        )),
    }
}

/// A call as chat completions writes one, in a message or a delta.
pub fn chat_tool_call_json(call: &ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments_json},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_parse_in_both_shapes() {
        assert_eq!(
            parse_choice(&json!("required"), "tool_choice").unwrap(),
            ToolChoice::Required
        );
        assert_eq!(
            parse_choice(
                &json!({"type": "function", "function": {"name": "f"}}),
                "tool_choice"
            )
            .unwrap(),
            ToolChoice::Function("f".into())
        );
        assert_eq!(
            parse_choice(&json!({"type": "function", "name": "f"}), "tool_choice").unwrap(),
            ToolChoice::Function("f".into())
        );
        assert!(parse_choice(&json!("sometimes"), "tool_choice").is_err());
        assert!(parse_choice(&json!({"type": "web_search"}), "tool_choice").is_err());
        assert!(parse_choice(&json!({"type": "function"}), "tool_choice").is_err());
    }

    #[test]
    fn arguments_must_be_an_object_in_a_string() {
        assert!(check_arguments(r#"{"city": "Paris"}"#, "messages").is_ok());
        assert!(check_arguments(r#""Paris""#, "messages").is_err());
        assert!(check_arguments("{city", "messages").is_err());
    }

    #[test]
    fn a_schema_that_is_not_an_object_is_refused() {
        let def = FunctionDef {
            name: "f".into(),
            description: None,
            parameters: Some(json!([1])),
            strict: None,
        };
        assert!(def.into_tool("tools").is_err());
    }
}
