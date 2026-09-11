//! The HTTP surface. Endpoints with no model worker behind them yet answer 501
//! rather than pretending.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::Request;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rkmodel_server_protocol::{Event, Input, Operation, Output, RkModelServer};
use serde_json::json;
use tokio_stream::StreamExt;

use crate::chat::{self, ChatRequest};
use crate::error::ApiError;
use crate::id;

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
        .route("/v1/chat/completions", post(chat_completions))
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

/// `generate` on the named model, streamed or not.
///
/// Everything refused here is refused before the daemon is touched, so a bad
/// request never reaches a model worker.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    request.validate()?;

    let model = request.model.clone();
    let streaming = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|o| o.include_usage);
    let input = Input::Generate(request.into_generate_input()?);

    let id = id::generate("chatcmpl-");
    let created = id::now();

    if !streaming {
        let outputs = state
            .daemon
            .invoke(Operation::Generate, &model, vec![input])
            .await?;
        let Some(Output::Generated {
            text,
            reasoning,
            finish,
            usage,
        }) = outputs.into_iter().next()
        else {
            return Err(ApiError::upstream("The daemon returned no generation."));
        };
        return Ok(Json(chat::completion_json(
            &id, created, &model, text, reasoning, finish, &usage,
        ))
        .into_response());
    }

    // From here the status line is already sent, so a failure is written as a
    // final event rather than an HTTP status.
    let mut events = state
        .daemon
        .invoke_stream(Operation::Generate, &model, input)
        .await?;

    let body = async_stream::stream! {
        let data = |value: serde_json::Value| {
            Ok::<_, Infallible>(SseEvent::default().data(value.to_string()))
        };

        yield data(chat::chunk_json(
            &id,
            created,
            &model,
            json!({"role": "assistant", "content": ""}),
            None,
        ));

        let mut done = None;
        while let Some(event) = events.next().await {
            match event {
                Ok(Event::ReasoningDelta(s)) => {
                    yield data(chat::chunk_json(&id, created, &model, json!({"reasoning_content": s}), None));
                }
                Ok(Event::TextDelta(s)) => {
                    yield data(chat::chunk_json(&id, created, &model, json!({"content": s}), None));
                }
                Ok(Event::Done { finish, usage }) => done = Some((finish, usage)),
                Ok(Event::Segment(_)) => {}
                Err(e) => {
                    yield data(ApiError::from(e).as_error_body());
                    return;
                }
            }
        }

        let Some((finish, usage)) = done else {
            // The run ended without a final event, which a cancelled run does.
            // Nothing more is owed to a client that has already gone.
            return;
        };

        yield data(chat::chunk_json(
            &id,
            created,
            &model,
            json!({}),
            Some(chat::finish_str(finish)),
        ));
        if include_usage {
            yield data(chat::usage_chunk_json(&id, created, &model, &usage));
        }
        yield Ok::<_, Infallible>(SseEvent::default().data("[DONE]"));
    };

    Ok(Sse::new(body).into_response())
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
    use rkmodel_server_protocol::{
        Error, EventStream, FinishReason, GenerateInput, ModelInfo, ModelState, Usage,
    };
    use std::sync::Mutex;
    use tower_service::Service as _;

    /// A daemon that replays scripted events, and records what it was asked
    /// for. Implements the trait, so none of this touches gRPC or a board.
    #[derive(Default)]
    struct Fake {
        models: Option<Vec<ModelInfo>>,
        outputs: Vec<Output>,
        events: Vec<Result<Event, Error>>,
        fail: Option<Error>,
        seen: Mutex<Vec<GenerateInput>>,
    }

    impl Fake {
        fn with_models(models: Vec<ModelInfo>) -> Fake {
            Fake {
                models: Some(models),
                ..Default::default()
            }
        }
        fn broken(error: Error) -> Fake {
            Fake {
                models: None,
                fail: Some(error),
                ..Default::default()
            }
        }
        fn answering(text: &str, reasoning: Option<&str>) -> Fake {
            Fake {
                models: Some(vec![]),
                outputs: vec![Output::Generated {
                    text: text.into(),
                    reasoning: reasoning.map(Into::into),
                    finish: FinishReason::Stop,
                    usage: Usage {
                        input_tokens: 24,
                        output_tokens: 388,
                        reasoning_tokens: 300,
                    },
                }],
                ..Default::default()
            }
        }
        fn streaming(events: Vec<Result<Event, Error>>) -> Fake {
            Fake {
                models: Some(vec![]),
                events,
                ..Default::default()
            }
        }
    }

    #[async_trait::async_trait]
    impl RkModelServer for Fake {
        async fn invoke(
            &self,
            _: Operation,
            _: &str,
            inputs: Vec<Input>,
        ) -> Result<Vec<Output>, Error> {
            if let Some(e) = &self.fail {
                return Err(e.clone());
            }
            for input in inputs {
                if let Input::Generate(g) = input {
                    self.seen.lock().unwrap().push(g);
                }
            }
            Ok(self.outputs.clone())
        }

        async fn invoke_stream(
            &self,
            _: Operation,
            _: &str,
            input: Input,
        ) -> Result<EventStream, Error> {
            if let Some(e) = &self.fail {
                return Err(e.clone());
            }
            if let Input::Generate(g) = input {
                self.seen.lock().unwrap().push(g);
            }
            let events = self.events.clone();
            Ok(Box::pin(tokio_stream::iter(events)))
        }

        async fn models(&self) -> Result<Vec<ModelInfo>, Error> {
            match &self.models {
                Some(m) => Ok(m.clone()),
                None => Err(self
                    .fail
                    .clone()
                    .unwrap_or_else(|| Error::Unreachable("down".into()))),
            }
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

    struct Harness {
        app: Router,
        daemon: Arc<Fake>,
    }

    fn harness(fake: Fake) -> Harness {
        let daemon = Arc::new(fake);
        let state = Arc::new(AppState {
            daemon: daemon.clone(),
            api_key: None,
        });
        Harness {
            app: router(state),
            daemon,
        }
    }

    impl Harness {
        async fn get(&mut self, uri: &str) -> (StatusCode, serde_json::Value) {
            let resp = self
                .app
                .call(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            let status = resp.status();
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            (status, serde_json::from_slice(&bytes).unwrap())
        }

        async fn post_raw(&mut self, uri: &str, body: serde_json::Value) -> (StatusCode, String) {
            let resp = self
                .app
                .call(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = resp.status();
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            (status, String::from_utf8(bytes.to_vec()).unwrap())
        }

        async fn post(
            &mut self,
            uri: &str,
            body: serde_json::Value,
        ) -> (StatusCode, serde_json::Value) {
            let (status, text) = self.post_raw(uri, body).await;
            (status, serde_json::from_str(&text).unwrap())
        }

        fn seen(&self) -> GenerateInput {
            self.daemon.seen.lock().unwrap()[0].clone()
        }
    }

    /// Pulls the JSON out of `data:` lines, in order. `[DONE]` becomes a null.
    fn sse_data(body: &str) -> Vec<serde_json::Value> {
        body.split("\n\n")
            .filter_map(|block| {
                let line = block.lines().find(|l| l.starts_with("data:"))?;
                let payload = line.trim_start_matches("data:").trim();
                if payload == "[DONE]" {
                    Some(serde_json::Value::Null)
                } else {
                    serde_json::from_str(payload).ok()
                }
            })
            .collect()
    }

    fn chat_body() -> serde_json::Value {
        json!({
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "why is the sky blue?"}]
        })
    }

    // ---- health and models -------------------------------------------------

    #[tokio::test]
    async fn health_is_ok_when_every_model_is_ready() {
        let mut h = harness(Fake::with_models(vec![info("qwen3-4b", ModelState::Ready)]));
        let (status, body) = h.get("/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["models"]["qwen3-4b"], "ready");
    }

    #[tokio::test]
    async fn health_says_why_when_a_model_is_not_ready() {
        let mut h = harness(Fake::with_models(vec![
            info("qwen3-4b", ModelState::Ready),
            info("whisper-small-30s", ModelState::Unavailable),
        ]));
        let (status, body) = h.get("/health").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["models"]["whisper-small-30s"], "unavailable");
    }

    #[tokio::test]
    async fn health_is_503_when_the_daemon_is_down() {
        let mut h = harness(Fake::broken(Error::Unreachable("connect refused".into())));
        let (status, body) = h.get("/health").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "unavailable");
    }

    #[tokio::test]
    async fn models_lists_what_the_daemon_reports() {
        let mut h = harness(Fake::with_models(vec![info("qwen3-4b", ModelState::Ready)]));
        let (status, body) = h.get("/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"][0]["id"], "qwen3-4b");
        assert_eq!(body["data"][0]["created"], 1757548800);
        assert_eq!(body["data"][0]["owned_by"], "local");
    }

    #[tokio::test]
    async fn a_down_daemon_becomes_503_in_openai_shape() {
        let mut h = harness(Fake::broken(Error::Unreachable("connect refused".into())));
        let (status, body) = h.get("/v1/models").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "server_error");
    }

    // ---- chat completions, non-streaming -----------------------------------

    #[tokio::test]
    async fn a_completion_has_the_documented_shape() {
        let mut h = harness(Fake::answering("the sky is blue", None));
        let (status, body) = h.post("/v1/chat/completions", chat_body()).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "chat.completion");
        assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(body["model"], "qwen3-4b");
        assert_eq!(body["choices"][0]["index"], 0);
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "the sky is blue");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 24);
        assert_eq!(body["usage"]["completion_tokens"], 388);
        assert_eq!(body["usage"]["total_tokens"], 412);
        assert_eq!(
            body["usage"]["completion_tokens_details"]["reasoning_tokens"],
            300
        );
    }

    #[tokio::test]
    async fn reasoning_content_is_absent_when_there_was_none() {
        let mut h = harness(Fake::answering("blue", None));
        let (_, body) = h.post("/v1/chat/completions", chat_body()).await;
        assert!(
            body["choices"][0]["message"]
                .get("reasoning_content")
                .is_none(),
            "{body}"
        );
    }

    #[tokio::test]
    async fn reasoning_content_carries_the_reasoning_when_there_was_some() {
        let mut h = harness(Fake::answering("blue", Some("scattering")));
        let (_, body) = h.post("/v1/chat/completions", chat_body()).await;
        assert_eq!(
            body["choices"][0]["message"]["reasoning_content"],
            "scattering"
        );
    }

    #[tokio::test]
    async fn a_length_finish_is_reported_as_length() {
        let mut h = harness(Fake {
            models: Some(vec![]),
            outputs: vec![Output::Generated {
                text: "cut off".into(),
                reasoning: None,
                finish: FinishReason::Length,
                usage: Usage::default(),
            }],
            ..Default::default()
        });
        let (_, body) = h.post("/v1/chat/completions", chat_body()).await;
        assert_eq!(body["choices"][0]["finish_reason"], "length");
    }

    // ---- request mapping ---------------------------------------------------

    #[tokio::test]
    async fn max_completion_tokens_wins_over_max_tokens() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = chat_body();
        body["max_tokens"] = json!(100);
        body["max_completion_tokens"] = json!(7);
        h.post("/v1/chat/completions", body).await;
        assert_eq!(h.seen().max_tokens, Some(7));
    }

    #[tokio::test]
    async fn max_tokens_is_used_when_it_is_the_only_one_sent() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = chat_body();
        body["max_tokens"] = json!(42);
        h.post("/v1/chat/completions", body).await;
        assert_eq!(h.seen().max_tokens, Some(42));
    }

    #[tokio::test]
    async fn sampling_fields_are_passed_through() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = chat_body();
        body["temperature"] = json!(0.2);
        body["top_p"] = json!(0.5);
        h.post("/v1/chat/completions", body).await;
        let seen = h.seen();
        assert_eq!(seen.temperature, Some(0.2));
        assert_eq!(seen.top_p, Some(0.5));
    }

    #[tokio::test]
    async fn an_absent_reasoning_effort_takes_the_model_default() {
        let mut h = harness(Fake::answering("x", None));
        h.post("/v1/chat/completions", chat_body()).await;
        assert_eq!(h.seen().reasoning, None);
    }

    #[tokio::test]
    async fn a_minimal_effort_turns_reasoning_off() {
        for effort in ["none", "minimal"] {
            let mut h = harness(Fake::answering("x", None));
            let mut body = chat_body();
            body["reasoning_effort"] = json!(effort);
            h.post("/v1/chat/completions", body).await;
            assert_eq!(h.seen().reasoning, Some(false), "effort {effort}");
        }
    }

    #[tokio::test]
    async fn any_other_effort_turns_reasoning_on() {
        for effort in ["low", "medium", "high"] {
            let mut h = harness(Fake::answering("x", None));
            let mut body = chat_body();
            body["reasoning_effort"] = json!(effort);
            h.post("/v1/chat/completions", body).await;
            assert_eq!(h.seen().reasoning, Some(true), "effort {effort}");
        }
    }

    #[tokio::test]
    async fn enable_thinking_from_a_vllm_client_is_honoured() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = chat_body();
        body["chat_template_kwargs"] = json!({"enable_thinking": false});
        body["reasoning_effort"] = json!("high");
        h.post("/v1/chat/completions", body).await;
        assert_eq!(
            h.seen().reasoning,
            Some(false),
            "the explicit kwarg wins over the effort"
        );
    }

    #[tokio::test]
    async fn a_developer_message_becomes_a_system_message() {
        let mut h = harness(Fake::answering("x", None));
        let body = json!({
            "model": "qwen3-4b",
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": "hi"}
            ]
        });
        h.post("/v1/chat/completions", body).await;
        let seen = h.seen();
        assert_eq!(seen.messages[0].role, rkmodel_server_protocol::Role::System);
        assert_eq!(seen.messages[1].role, rkmodel_server_protocol::Role::User);
    }

    #[tokio::test]
    async fn content_as_an_array_of_text_parts_is_joined() {
        let mut h = harness(Fake::answering("x", None));
        let body = json!({
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "one "},
                {"type": "text", "text": "two"}
            ]}]
        });
        h.post("/v1/chat/completions", body).await;
        assert_eq!(h.seen().messages[0].parts.len(), 2);
    }

    #[tokio::test]
    async fn hints_are_accepted_and_ignored() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = chat_body();
        body["user"] = json!("someone");
        body["metadata"] = json!({"a": "b"});
        body["store"] = json!(true);
        body["seed"] = json!(7);
        body["parallel_tool_calls"] = json!(false);
        body["logprobs"] = json!(false);
        let (status, _) = h.post("/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK);
    }

    // ---- validation --------------------------------------------------------

    async fn refused(field: &str, patch: serde_json::Value) {
        let mut h = harness(Fake::answering("x", None));
        let mut body = chat_body();
        for (k, v) in patch.as_object().unwrap() {
            body[k] = v.clone();
        }
        let (status, response) = h.post("/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert_eq!(response["error"]["type"], "invalid_request_error");
        assert_eq!(response["error"]["param"], field, "{response}");
    }

    #[tokio::test]
    async fn more_than_one_choice_is_refused() {
        refused("n", json!({"n": 2})).await;
    }

    #[tokio::test]
    async fn tools_are_refused() {
        refused("tools", json!({"tools": [{"type": "function"}]})).await;
        refused("tool_choice", json!({"tool_choice": "auto"})).await;
    }

    #[tokio::test]
    async fn logprobs_are_refused_when_asked_for() {
        refused("logprobs", json!({"logprobs": true})).await;
    }

    #[tokio::test]
    async fn a_structured_response_format_is_refused() {
        refused(
            "response_format",
            json!({"response_format": {"type": "json_object"}}),
        )
        .await;
    }

    #[tokio::test]
    async fn a_plain_text_response_format_is_allowed() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = chat_body();
        body["response_format"] = json!({"type": "text"});
        let (status, _) = h.post("/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn an_unknown_role_is_refused() {
        refused(
            "messages",
            json!({"messages": [{"role": "wizard", "content": "hi"}]}),
        )
        .await;
    }

    #[tokio::test]
    async fn no_messages_is_refused() {
        refused("messages", json!({"messages": []})).await;
    }

    #[tokio::test]
    async fn an_unknown_model_becomes_404_with_model_not_found() {
        let mut h = harness(Fake::broken(Error::UnknownModel("nope".into())));
        let (status, body) = h.post("/v1/chat/completions", chat_body()).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    async fn a_busy_daemon_becomes_503_with_retry_after() {
        let daemon = Arc::new(Fake::broken(Error::Busy {
            retry_after_ms: 1500,
        }));
        let state = Arc::new(AppState {
            daemon: daemon.clone(),
            api_key: None,
        });
        let mut app = router(state);
        let resp = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(chat_body().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            "2",
            "1500ms rounds up to whole seconds"
        );
    }

    // ---- chat completions, streaming ---------------------------------------

    fn streaming_body(include_usage: bool) -> serde_json::Value {
        let mut body = chat_body();
        body["stream"] = json!(true);
        if include_usage {
            body["stream_options"] = json!({"include_usage": true});
        }
        body
    }

    #[tokio::test]
    async fn a_stream_follows_the_documented_sequence() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("the sky ".into())),
            Ok(Event::TextDelta("is blue".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage {
                    input_tokens: 24,
                    output_tokens: 2,
                    reasoning_tokens: 0,
                },
            }),
        ]));
        let (status, body) = h
            .post_raw("/v1/chat/completions", streaming_body(false))
            .await;
        assert_eq!(status, StatusCode::OK);

        let chunks = sse_data(&body);
        // role chunk, two deltas, the finish chunk, then [DONE]
        assert_eq!(chunks.len(), 5, "{body}");
        assert_eq!(chunks[0]["object"], "chat.completion.chunk");
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "");
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "the sky ");
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "is blue");
        assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
        assert_eq!(chunks[3]["choices"][0]["delta"], json!({}));
        assert_eq!(chunks[4], serde_json::Value::Null, "ends with [DONE]");
    }

    #[tokio::test]
    async fn every_streamed_chunk_shares_one_id() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("a".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/chat/completions", streaming_body(false))
            .await;
        let chunks = sse_data(&body);
        let ids: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.get("id").and_then(|v| v.as_str()))
            .collect();
        assert!(ids.len() >= 3);
        assert!(ids.windows(2).all(|w| w[0] == w[1]), "{ids:?}");
    }

    #[tokio::test]
    async fn reasoning_deltas_stream_on_their_own_field() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::ReasoningDelta("hmm".into())),
            Ok(Event::TextDelta("blue".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/chat/completions", streaming_body(false))
            .await;
        let chunks = sse_data(&body);
        assert_eq!(chunks[1]["choices"][0]["delta"]["reasoning_content"], "hmm");
        assert!(chunks[1]["choices"][0]["delta"].get("content").is_none());
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "blue");
    }

    #[tokio::test]
    async fn include_usage_adds_a_final_usage_chunk_with_no_choices() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("a".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage {
                    input_tokens: 24,
                    output_tokens: 1,
                    reasoning_tokens: 0,
                },
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/chat/completions", streaming_body(true))
            .await;
        let chunks = sse_data(&body);
        let usage = &chunks[chunks.len() - 2];
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["prompt_tokens"], 24);
        assert_eq!(usage["usage"]["total_tokens"], 25);
        assert_eq!(chunks[chunks.len() - 1], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn without_include_usage_no_usage_chunk_is_sent() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("a".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/chat/completions", streaming_body(false))
            .await;
        assert!(!body.contains("\"usage\""), "{body}");
    }

    #[tokio::test]
    async fn a_failure_partway_through_becomes_a_final_error_event() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("the sky ".into())),
            Err(Error::Runtime {
                call: "run_llm".into(),
                code: -1,
            }),
        ]));
        let (status, body) = h
            .post_raw("/v1/chat/completions", streaming_body(false))
            .await;
        // The status line went out with the first chunk, so this cannot be a 500.
        assert_eq!(status, StatusCode::OK);
        let chunks = sse_data(&body);
        let last = chunks.last().unwrap();
        assert_eq!(last["error"]["type"], "server_error", "{body}");
        assert!(!body.contains("[DONE]"), "a failed stream is not completed");
    }

    #[tokio::test]
    async fn a_stream_that_ends_without_done_just_stops() {
        // What a cancelled run looks like: deltas, then nothing.
        let mut h = harness(Fake::streaming(vec![Ok(Event::TextDelta("a".into()))]));
        let (status, body) = h
            .post_raw("/v1/chat/completions", streaming_body(false))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert!(!body.contains("[DONE]"), "{body}");
        assert!(!body.contains("finish_reason\":\"stop"), "{body}");
    }

    #[tokio::test]
    async fn a_failure_before_the_stream_starts_is_still_an_http_status() {
        let mut h = harness(Fake::broken(Error::UnknownModel("nope".into())));
        let (status, body) = h.post("/v1/chat/completions", streaming_body(false)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
    }

    // ---- still unimplemented ----------------------------------------------

    #[tokio::test]
    async fn the_other_endpoints_still_answer_501() {
        for uri in [
            "/v1/responses",
            "/v1/audio/transcriptions",
            "/v1/embeddings",
        ] {
            let mut h = harness(Fake::answering("x", None));
            let (status, _) = h.post(uri, json!({})).await;
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{uri}");
        }
    }
}
