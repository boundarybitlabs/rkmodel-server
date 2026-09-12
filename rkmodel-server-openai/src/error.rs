//! OpenAI's error body, so SDKs raise their usual exception types and retry the
//! 5xx ones.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use rkmodel_server_protocol::Error;
use serde_json::json;

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    body: serde_json::Value,
    retry_after_ms: Option<u32>,
}

impl ApiError {
    /// A request the frontend refuses on its own, naming the offending field.
    /// This is what the validation table in the design calls for, and it has no
    /// caller until the first endpoint that parses a body lands.
    #[allow(dead_code)]
    pub fn invalid_request(message: impl Into<String>, param: Option<&str>) -> ApiError {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            body: json!({"error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "param": param,
                "code": null,
            }}),
            retry_after_ms: None,
        }
    }

    /// An upload refused on size. OpenAI answers 413 for this, and SDKs map
    /// that to their own error rather than the generic 400.
    pub fn too_large(message: impl Into<String>) -> ApiError {
        ApiError {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            body: json!({"error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "param": "file",
                "code": null,
            }}),
            retry_after_ms: None,
        }
    }

    pub fn not_implemented(message: impl Into<String>) -> ApiError {
        ApiError {
            status: StatusCode::NOT_IMPLEMENTED,
            body: json!({"error": {
                "message": message.into(),
                "type": "server_error",
                "param": null,
                "code": null,
            }}),
            retry_after_ms: None,
        }
    }

    /// The daemon answered, but not with anything usable.
    pub fn upstream(message: impl Into<String>) -> ApiError {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: json!({"error": {
                "message": message.into(),
                "type": "server_error",
                "param": null,
                "code": null,
            }}),
            retry_after_ms: None,
        }
    }

    /// The error body on its own, for a failure that lands mid-stream where the
    /// status line has already gone out.
    pub fn as_error_body(&self) -> serde_json::Value {
        self.body.clone()
    }

    pub fn unauthorized() -> ApiError {
        ApiError {
            status: StatusCode::UNAUTHORIZED,
            body: json!({"error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_api_key",
            }}),
            retry_after_ms: None,
        }
    }
}

impl From<Error> for ApiError {
    fn from(e: Error) -> ApiError {
        let m = e.http();
        ApiError {
            status: StatusCode::from_u16(m.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            body: json!({"error": {
                "message": e.to_string(),
                "type": m.kind,
                "param": null,
                "code": m.code,
            }}),
            retry_after_ms: m.retry_after_ms,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut resp = (self.status, Json(self.body)).into_response();
        if let Some(ms) = self.retry_after_ms {
            // Retry-After is whole seconds, and zero would invite a hot loop.
            let seconds = ms.div_ceil(1000).max(1);
            if let Ok(v) = HeaderValue::from_str(&seconds.to_string()) {
                resp.headers_mut().insert(header::RETRY_AFTER, v);
            }
        }
        resp
    }
}
