//! The HTTP surface. Only the endpoints that need no model worker are wired;
//! the rest answer 501 rather than pretending.

use std::sync::Arc;

use axum::extract::State;
use axum::http::Request;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use rkmodel_server_protocol::RkModelServer;
use serde_json::json;

use crate::error::ApiError;

pub struct AppState {
    pub daemon: Arc<dyn RkModelServer>,
    /// Checked against `Authorization: Bearer` when set. Separate from the
    /// token the daemon uses.
    pub api_key: Option<String>,
}

pub fn router(state: Arc<AppState>) -> Router {
    // `/health` sits outside the key check, so a monitor can reach it without
    // holding a credential.
    let v1 = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(not_yet))
        .route("/v1/responses", post(not_yet))
        .route("/v1/audio/transcriptions", post(not_yet))
        .route("/v1/embeddings", post(not_yet))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ))
        .with_state(state.clone());

    Router::new()
        .route("/health", get(health))
        .with_state(state)
        .merge(v1)
}

/// Checks `Authorization: Bearer` when an API key is configured. That key is
/// separate from the token the daemon uses.
async fn require_api_key(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    let Some(expected) = &state.api_key else {
        return Ok(next.run(request).await);
    };
    let presented = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    match presented {
        Some(got) if got == expected => Ok(next.run(request).await),
        _ => Err(ApiError::unauthorized()),
    }
}

/// 200 when the daemon is reachable and every configured model is ready, 503
/// otherwise. The body always lists each model's state, so a 503 says why.
async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.daemon.models().await {
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unavailable", "detail": e.to_string(), "models": {}})),
        ),
        Ok(models) => {
            let all_ready = !models.is_empty() && models.iter().all(|m| m.state.is_ready());
            let states: serde_json::Map<String, serde_json::Value> = models
                .iter()
                .map(|m| (m.id.clone(), json!(m.state.as_str())))
                .collect();
            let status = if all_ready {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            (
                status,
                Json(json!({
                    "status": if all_ready { "ok" } else { "degraded" },
                    "models": states,
                })),
            )
        }
    }
}

/// From the daemon's `models()`. `created` is when the daemon loaded the model.
async fn models(State(state): State<Arc<AppState>>) -> Result<Json<serde_json::Value>, ApiError> {
    let models = state.daemon.models().await?;
    let data: Vec<_> = models
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "object": "model",
                "created": m.loaded_at,
                "owned_by": "local",
            })
        })
        .collect();
    Ok(Json(json!({"object": "list", "data": data})))
}

async fn not_yet() -> ApiError {
    ApiError::not_implemented(
        "This endpoint is not implemented yet. See the milestones in MODEL_SERVER.md.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use rkmodel_server_protocol::{
        Error, Event, EventStream, Input, ModelInfo, ModelState, Operation, Output,
    };
    use tower_service::Service as _;

    /// The fake daemon the frontend tests against. It implements the trait, so
    /// none of this touches gRPC or a board.
    struct Fake(Result<Vec<ModelInfo>, Error>);

    #[async_trait::async_trait]
    impl RkModelServer for Fake {
        async fn invoke(&self, _: Operation, _: &str, _: Vec<Input>) -> Result<Vec<Output>, Error> {
            unimplemented!()
        }
        async fn invoke_stream(
            &self,
            _: Operation,
            _: &str,
            _: Input,
        ) -> Result<EventStream, Error> {
            let _ = Event::TextDelta(String::new());
            unimplemented!()
        }
        async fn models(&self) -> Result<Vec<ModelInfo>, Error> {
            self.0.clone()
        }
    }

    fn info(id: &str, state: ModelState) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            operations: vec![Operation::Generate],
            state,
            loaded_at: 1757548800,
            image_input: None,
            reasoning: false,
        }
    }

    async fn call(fake: Fake, uri: &str) -> (StatusCode, serde_json::Value) {
        let state = Arc::new(AppState {
            daemon: Arc::new(fake),
            api_key: None,
        });
        let mut app = router(state);
        let resp = app
            .call(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn health_is_ok_when_every_model_is_ready() {
        let (status, body) = call(
            Fake(Ok(vec![info("qwen3-4b", ModelState::Ready)])),
            "/health",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["models"]["qwen3-4b"], "ready");
    }

    #[tokio::test]
    async fn health_says_why_when_a_model_is_not_ready() {
        let (status, body) = call(
            Fake(Ok(vec![
                info("qwen3-4b", ModelState::Ready),
                info("whisper-small-30s", ModelState::Unavailable),
            ])),
            "/health",
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["models"]["whisper-small-30s"], "unavailable");
    }

    #[tokio::test]
    async fn health_is_503_when_the_daemon_is_down() {
        let (status, body) = call(
            Fake(Err(Error::Unreachable("connect refused".into()))),
            "/health",
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "unavailable");
    }

    #[tokio::test]
    async fn models_lists_what_the_daemon_reports() {
        let (status, body) = call(
            Fake(Ok(vec![info("qwen3-4b", ModelState::Ready)])),
            "/v1/models",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"][0]["id"], "qwen3-4b");
        assert_eq!(body["data"][0]["created"], 1757548800);
        assert_eq!(body["data"][0]["owned_by"], "local");
    }

    #[tokio::test]
    async fn a_down_daemon_becomes_503_in_openai_shape() {
        let (status, body) = call(
            Fake(Err(Error::Unreachable("connect refused".into()))),
            "/v1/models",
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "server_error");
    }
}
