//! The HTTP surface. Endpoints with no model worker behind them yet answer 501
//! rather than pretending.

use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::header;
use axum::http::Request;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_extra::extract::Multipart;
use rkmodel_server_protocol::{
    transcript, Event, Input, Operation, Output, RkModelServer, TranscribeInput,
};
use serde_json::json;
use tokio_stream::StreamExt;

use crate::audio::{self, decode};
use crate::chat::{self, ChatRequest};
use crate::error::ApiError;
use crate::id;
use crate::responses::{self, ResponsesRequest};
use crate::tools;

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
        .route("/v1/responses", post(responses_endpoint))
        // Axum's default body limit is 2 MB, well under the upload limit this
        // endpoint documents, so it is raised for this route alone.
        .route(
            "/v1/audio/transcriptions",
            post(transcriptions).layer(DefaultBodyLimit::max(audio::MAX_BODY_BYTES)),
        )
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
            tool_calls,
            finish,
            usage,
        }) = outputs.into_iter().next()
        else {
            return Err(ApiError::upstream("The daemon returned no generation."));
        };
        return Ok(Json(chat::completion_json(
            &id,
            created,
            &model,
            text,
            reasoning,
            &tool_calls,
            finish,
            &usage,
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
        let mut calls = 0usize;
        while let Some(event) = events.next().await {
            match event {
                Ok(Event::ReasoningDelta(s)) => {
                    yield data(chat::chunk_json(&id, created, &model, json!({"reasoning_content": s}), None));
                }
                Ok(Event::TextDelta(s)) => {
                    yield data(chat::chunk_json(&id, created, &model, json!({"content": s}), None));
                }
                Ok(Event::ToolCall(call)) => {
                    // Each call arrives whole, so its one delta carries the id,
                    // the name and every argument at once.
                    let mut delta = tools::chat_tool_call_json(&call);
                    delta["index"] = json!(calls);
                    calls += 1;
                    yield data(chat::chunk_json(&id, created, &model, json!({"tool_calls": [delta]}), None));
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

/// `generate` on the named model, in the Responses API's shapes.
///
/// The same `GenerateInput` chat completions builds. Only the parsing and the
/// emitted shapes differ.
async fn responses_endpoint(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ResponsesRequest>,
) -> Result<Response, ApiError> {
    request.validate()?;

    let model = request.model.clone();
    let streaming = request.stream;
    let input = Input::Generate(request.into_generate_input()?);

    let ids = responses::Ids::new();
    let created = id::now();

    if !streaming {
        let outputs = state
            .daemon
            .invoke(Operation::Generate, &model, vec![input])
            .await?;
        let Some(Output::Generated {
            text,
            reasoning,
            tool_calls,
            finish,
            usage,
        }) = outputs.into_iter().next()
        else {
            return Err(ApiError::upstream("The daemon returned no generation."));
        };
        return Ok(Json(responses::completed_json(
            &ids,
            created,
            &model,
            &text,
            reasoning.as_deref(),
            &tool_calls,
            finish,
            &usage,
        ))
        .into_response());
    }

    let mut events = state
        .daemon
        .invoke_stream(Operation::Generate, &model, input)
        .await?;

    let body = async_stream::stream! {
        let mut seq: u64 = 0;
        // Every event carries its own type and a sequence number, and the
        // stream ends on a terminal event rather than a [DONE] sentinel.
        macro_rules! ev {
            ($name:expr, $body:expr) => {{
                let mut value = $body;
                value["type"] = json!($name);
                value["sequence_number"] = json!(seq);
                seq += 1;
                Ok::<_, Infallible>(SseEvent::default().event($name).data(value.to_string()))
            }};
        }

        let in_progress = |ids: &responses::Ids| {
            responses::response_json(ids, created, &model, "in_progress", vec![], None, None)
        };
        yield ev!("response.created", json!({"response": in_progress(&ids)}));
        yield ev!("response.in_progress", json!({"response": in_progress(&ids)}));

        let mut items = responses::StreamState::new(ids);
        let mut done: Option<(rkmodel_server_protocol::FinishReason, rkmodel_server_protocol::Usage)> = None;
        let mut failed = None;

        while let Some(event) = events.next().await {
            let named = match event {
                Ok(Event::ReasoningDelta(delta)) => items.reasoning_delta(&delta),
                Ok(Event::TextDelta(delta)) => items.text_delta(&delta),
                Ok(Event::ToolCall(call)) => items.tool_call(&call),
                Ok(Event::Done { finish, usage }) => {
                    done = Some((finish, usage));
                    Vec::new()
                }
                Ok(Event::Segment(_)) => Vec::new(),
                Err(e) => {
                    failed = Some(ApiError::from(e));
                    break;
                }
            };
            for (name, body) in named {
                yield ev!(name, body);
            }
        }

        if let Some(error) = failed {
            yield ev!("response.failed", json!({"error": error.as_error_body()["error"]}));
            // The counter is finished with; naming it keeps the last bump read.
            let _ = seq;
            return;
        }

        let Some((finish, usage)) = done else {
            // A cancelled run ends without a final event. Nothing more is owed
            // to a client that has already gone.
            return;
        };

        for (name, body) in items.finish() {
            yield ev!(name, body);
        }
        let final_body = responses::response_json(
            items.ids(),
            created,
            &model,
            responses::status_for(finish),
            items.output(),
            Some(finish),
            Some(&usage),
        );
        yield ev!(responses::terminal_event(finish), json!({"response": final_body}));
        let _ = seq;
    };

    Ok(Sse::new(body).into_response())
}

/// `transcribe` on the named model.
///
/// The upload is decoded here and reaches the daemon as a stream, so rkwhisperd
/// starts on the first second of a recording while the rest is still being
/// decoded, and a long clip is never held in memory as PCM.
async fn transcriptions(
    State(state): State<Arc<AppState>>,
    form: Multipart,
) -> Result<Response, ApiError> {
    let request = audio::parse(form).await?;

    // Probing reads only the container's headers. Doing it before the call
    // means an unreadable upload is a 400 rather than a stream the daemon has
    // already begun reading when it fails.
    let source = decode::Source::open(request.file, request.filename.as_deref())
        .map_err(|e| ApiError::invalid_request(e.to_string(), Some("file")))?;
    tracing::debug!(
        model = %request.model,
        rate = source.rate(),
        "decoding an upload for transcription"
    );

    let (pcm_s16le, produced) = source.into_stream();
    let input = Input::Transcribe(TranscribeInput {
        pcm_s16le,
        language: request.language.clone(),
    });

    let mut events = state
        .daemon
        .invoke_stream(Operation::Transcribe, &request.model, input)
        .await?;

    // Transcription is not streamed to the caller yet, so the segments are
    // collected here and written as one body. rkwhisperd sends them as it
    // decodes, which is what a future `stream: true` would forward.
    let mut segments = Vec::new();
    let mut done = false;
    while let Some(event) = events.next().await {
        match event? {
            Event::Segment(segment) => segments.push(segment),
            Event::Done { .. } => done = true,
            Event::TextDelta(_) | Event::ReasoningDelta(_) | Event::ToolCall(_) => {}
        }
    }
    if !done {
        return Err(ApiError::upstream(
            "The daemon ended the transcription without a final event.",
        ));
    }

    // The decoder counted what it produced, so this is the clip the daemon was
    // actually sent rather than whatever the upload's header claimed.
    let duration_s = produced.load(Ordering::Relaxed) as f32 / decode::TARGET_RATE as f32;
    let body = audio::body(
        request.format,
        &transcript(&segments),
        &segments,
        duration_s,
        request.language.as_deref(),
    );

    Ok((
        [(header::CONTENT_TYPE, request.format.content_type())],
        body,
    )
        .into_response())
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
        Error, EventStream, FinishReason, GenerateInput, ModelInfo, ModelState, Part, Role,
        Segment, ToolCall, ToolChoice, Usage,
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
        /// The language and the PCM a transcribe call carried, once its audio
        /// stream has been drained the way the real daemon drains it.
        heard: Mutex<Vec<(Option<String>, Vec<u8>)>>,
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
                    tool_calls: vec![],
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
            match input {
                Input::Generate(g) => self.seen.lock().unwrap().push(g),
                Input::Transcribe(t) => {
                    // Draining matters: the frontend's decoder writes into a
                    // bounded channel, so a daemon that never reads would stall
                    // it rather than fail a test loudly.
                    let mut audio = t.pcm_s16le;
                    let mut pcm = Vec::new();
                    while let Some(chunk) = audio.next().await {
                        pcm.extend_from_slice(&chunk?);
                    }
                    self.heard.lock().unwrap().push((t.language, pcm));
                }
                Input::Embed { .. } => {}
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
            tools: false,
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

        /// Posts a multipart body built by hand, so these tests need no client.
        async fn post_form(
            &mut self,
            fields: &[(&str, &str)],
            file: Option<(&str, &[u8])>,
        ) -> (StatusCode, String, Option<String>) {
            const BOUNDARY: &str = "----rkmodelservertest";
            let mut body: Vec<u8> = Vec::new();
            for (name, value) in fields {
                body.extend_from_slice(
                    format!(
                        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                    )
                    .as_bytes(),
                );
            }
            if let Some((filename, bytes)) = file {
                body.extend_from_slice(
                    format!(
                        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; \
                         filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                    )
                    .as_bytes(),
                );
                body.extend_from_slice(bytes);
                body.extend_from_slice(b"\r\n");
            }
            body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());

            let resp = self
                .app
                .call(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/audio/transcriptions")
                        .header(
                            "content-type",
                            format!("multipart/form-data; boundary={BOUNDARY}"),
                        )
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = resp.status();
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            (
                status,
                String::from_utf8(bytes.to_vec()).unwrap(),
                content_type,
            )
        }

        fn heard(&self) -> (Option<String>, Vec<u8>) {
            self.daemon.heard.lock().unwrap()[0].clone()
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
                tool_calls: vec![],
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

    // ---- tools on chat completions -----------------------------------------

    fn weather_call(id: &str, city: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "get_weather".into(),
            arguments_json: format!(r#"{{"city":"{city}"}}"#),
        }
    }

    fn calling(calls: Vec<ToolCall>) -> Fake {
        Fake {
            models: Some(vec![]),
            outputs: vec![Output::Generated {
                text: String::new(),
                reasoning: None,
                tool_calls: calls,
                finish: FinishReason::ToolCalls,
                usage: Usage::default(),
            }],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn tools_and_a_tool_loop_reach_the_daemon() {
        let mut h = harness(calling(vec![]));
        let body = json!({
            "model": "qwen3-4b",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": null, "reasoning_content": "need it",
                 "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "21C"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather",
                "description": "Current weather", "strict": true,
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "parallel_tool_calls": false,
        });
        let (status, response) = h.post("/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "{response}");

        let seen = h.daemon.seen.lock().unwrap()[0].clone();
        assert_eq!(seen.tools.len(), 1);
        assert_eq!(
            seen.tools[0].parameters_json.as_deref(),
            Some(r#"{"type":"object","properties":{"city":{"type":"string"}}}"#)
        );
        assert_eq!(
            seen.tool_choice,
            Some(ToolChoice::Function("get_weather".into()))
        );
        assert_eq!(seen.parallel_tool_calls, Some(false));
        assert_eq!(seen.messages[1].reasoning.as_deref(), Some("need it"));
        assert_eq!(seen.messages[1].tool_calls[0].id, "call_1");
        assert_eq!(seen.messages[2].role, Role::Tool);
        assert_eq!(seen.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[tokio::test]
    async fn a_completion_that_calls_tools_has_null_content() {
        let mut h = harness(calling(vec![
            weather_call("call_a", "Paris"),
            weather_call("call_b", "London"),
        ]));
        let (_, body) = h.post("/v1/chat/completions", chat_body()).await;
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], serde_json::Value::Null);
        assert_eq!(message["tool_calls"][1]["id"], "call_b");
        assert_eq!(message["tool_calls"][1]["type"], "function");
        assert_eq!(message["tool_calls"][1]["function"]["name"], "get_weather");
        assert_eq!(
            message["tool_calls"][1]["function"]["arguments"],
            r#"{"city":"London"}"#
        );
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn a_streamed_call_is_one_delta_at_its_index() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::ToolCall(weather_call("call_a", "Paris"))),
            Ok(Event::ToolCall(weather_call("call_b", "London"))),
            Ok(Event::Done {
                finish: FinishReason::ToolCalls,
                usage: Usage::default(),
            }),
        ]));
        let mut body = chat_body();
        body["stream"] = json!(true);
        let (_, text) = h.post_raw("/v1/chat/completions", body).await;
        let chunks = sse_data(&text);
        let calls: Vec<_> = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array())
            .collect();
        assert_eq!(calls.len(), 2, "{text}");
        assert_eq!(calls[0][0]["index"], 0);
        assert_eq!(calls[1][0]["index"], 1);
        assert_eq!(calls[1][0]["id"], "call_b");
        assert_eq!(calls[1][0]["function"]["arguments"], r#"{"city":"London"}"#);
        assert!(text.contains(r#""finish_reason":"tool_calls""#), "{text}");
    }

    #[tokio::test]
    async fn tool_requests_it_cannot_serve_are_refused() {
        refused("tools", json!({"tools": [{"type": "web_search"}]})).await;
        refused(
            "tools",
            json!({"tools": [{"type": "function", "function": {"name": "f", "parameters": [1]}}]}),
        )
        .await;
        refused("tool_choice", json!({"tool_choice": "sometimes"})).await;
        refused("functions", json!({"functions": [{"name": "f"}]})).await;
        refused("function_call", json!({"function_call": "auto"})).await;
        refused(
            "messages",
            json!({"messages": [{"role": "tool", "content": "21C"}]}),
        )
        .await;
        refused(
            "messages",
            json!({"messages": [{"role": "assistant", "tool_calls": [{"id": "c", "type": "function",
                "function": {"name": "f", "arguments": "\"Paris\""}}]}]}),
        )
        .await;
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

    // ---- responses, non-streaming -----------------------------------------

    fn responses_body() -> serde_json::Value {
        json!({"model": "qwen3-4b", "input": "why is the sky blue?"})
    }

    #[tokio::test]
    async fn a_response_has_the_documented_shape() {
        let mut h = harness(Fake::answering("the sky is blue", None));
        let (status, body) = h.post("/v1/responses", responses_body()).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "response");
        assert!(body["id"].as_str().unwrap().starts_with("resp_"));
        assert_eq!(body["status"], "completed");
        assert_eq!(body["model"], "qwen3-4b");
        assert_eq!(body["incomplete_details"], serde_json::Value::Null);

        let output = body["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "no reasoning item when there was none");
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["role"], "assistant");
        assert_eq!(output[0]["status"], "completed");
        assert_eq!(output[0]["content"][0]["type"], "output_text");
        assert_eq!(output[0]["content"][0]["text"], "the sky is blue");
        assert_eq!(output[0]["content"][0]["annotations"], json!([]));

        assert_eq!(body["usage"]["input_tokens"], 24);
        assert_eq!(body["usage"]["output_tokens"], 388);
        assert_eq!(body["usage"]["total_tokens"], 412);
        assert_eq!(
            body["usage"]["output_tokens_details"]["reasoning_tokens"],
            300
        );
    }

    #[tokio::test]
    async fn a_reasoning_item_leads_the_output_when_the_model_reasoned() {
        let mut h = harness(Fake::answering("blue", Some("scattering")));
        let (_, body) = h.post("/v1/responses", responses_body()).await;
        let output = body["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert!(output[0]["id"].as_str().unwrap().starts_with("rs_"));
        assert_eq!(output[0]["summary"], json!([]));
        assert_eq!(output[0]["content"][0]["type"], "reasoning_text");
        assert_eq!(output[0]["content"][0]["text"], "scattering");
        assert_eq!(output[1]["type"], "message");
        assert!(output[1]["id"].as_str().unwrap().starts_with("msg_"));
    }

    #[tokio::test]
    async fn a_length_finish_becomes_incomplete_with_a_reason() {
        let mut h = harness(Fake {
            models: Some(vec![]),
            outputs: vec![Output::Generated {
                text: "cut off".into(),
                reasoning: None,
                tool_calls: vec![],
                finish: FinishReason::Length,
                usage: Usage::default(),
            }],
            ..Default::default()
        });
        let (_, body) = h.post("/v1/responses", responses_body()).await;
        assert_eq!(body["status"], "incomplete");
        assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
    }

    // ---- responses, request mapping ---------------------------------------

    #[tokio::test]
    async fn a_string_input_becomes_one_user_message() {
        let mut h = harness(Fake::answering("x", None));
        h.post("/v1/responses", responses_body()).await;
        let seen = h.seen();
        assert_eq!(seen.messages.len(), 1);
        assert_eq!(seen.messages[0].role, rkmodel_server_protocol::Role::User);
    }

    #[tokio::test]
    async fn instructions_become_a_system_message_placed_first() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = responses_body();
        body["instructions"] = json!("be brief");
        h.post("/v1/responses", body).await;
        let seen = h.seen();
        assert_eq!(seen.messages[0].role, rkmodel_server_protocol::Role::System);
        assert_eq!(seen.messages[1].role, rkmodel_server_protocol::Role::User);
    }

    #[tokio::test]
    async fn message_items_carry_their_roles_and_parts() {
        let mut h = harness(Fake::answering("x", None));
        let body = json!({
            "model": "qwen3-4b",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "reply"}]},
                {"role": "user", "content": "second"}
            ]
        });
        h.post("/v1/responses", body).await;
        let seen = h.seen();
        assert_eq!(seen.messages.len(), 3);
        assert_eq!(
            seen.messages[1].role,
            rkmodel_server_protocol::Role::Assistant
        );
    }

    #[tokio::test]
    async fn reasoning_items_echoed_back_are_ignored() {
        let mut h = harness(Fake::answering("x", None));
        let body = json!({
            "model": "qwen3-4b",
            "input": [
                {"type": "reasoning", "id": "rs_1", "summary": [],
                 "content": [{"type": "reasoning_text", "text": "old thinking"}]},
                {"type": "message", "role": "user", "content": "hi"}
            ]
        });
        h.post("/v1/responses", body).await;
        let seen = h.seen();
        assert_eq!(seen.messages.len(), 1, "only the user turn survives");
        assert_eq!(seen.messages[0].role, rkmodel_server_protocol::Role::User);
    }

    #[tokio::test]
    async fn max_output_tokens_is_the_budget() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = responses_body();
        body["max_output_tokens"] = json!(33);
        h.post("/v1/responses", body).await;
        assert_eq!(h.seen().max_tokens, Some(33));
    }

    #[tokio::test]
    async fn reasoning_effort_maps_the_same_way_as_chat() {
        for (effort, want) in [
            ("none", Some(false)),
            ("minimal", Some(false)),
            ("low", Some(true)),
            ("high", Some(true)),
        ] {
            let mut h = harness(Fake::answering("x", None));
            let mut body = responses_body();
            body["reasoning"] = json!({"effort": effort});
            h.post("/v1/responses", body).await;
            assert_eq!(h.seen().reasoning, want, "effort {effort}");
        }
    }

    // ---- responses, validation --------------------------------------------

    async fn responses_refused(field: &str, patch: serde_json::Value) {
        let mut h = harness(Fake::answering("x", None));
        let mut body = responses_body();
        for (k, v) in patch.as_object().unwrap() {
            body[k] = v.clone();
        }
        let (status, response) = h.post("/v1/responses", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert_eq!(response["error"]["param"], field, "{response}");
    }

    #[tokio::test]
    async fn continuing_a_stored_response_is_refused() {
        responses_refused(
            "previous_response_id",
            json!({"previous_response_id": "resp_1"}),
        )
        .await;
    }

    // ---- tools on responses ------------------------------------------------

    #[tokio::test]
    async fn function_call_items_fold_into_one_assistant_turn() {
        let mut h = harness(calling(vec![]));
        let body = json!({
            "model": "qwen3-4b",
            "input": [
                {"role": "user", "content": "Weather in Paris and London?"},
                {"type": "reasoning", "id": "rs_1", "summary": [],
                 "content": [{"type": "reasoning_text", "text": "need both"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
                {"type": "function_call", "id": "fc_2", "call_id": "call_2",
                 "name": "get_weather", "arguments": "{\"city\": \"London\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "21C"},
                {"type": "function_call_output", "call_id": "call_2",
                 "output": [{"type": "input_text", "text": "15C"}]},
            ],
            "tools": [{"type": "function", "name": "get_weather", "description": "Current weather",
                "parameters": {"type": "object"}, "strict": true}],
            "tool_choice": {"type": "function", "name": "get_weather"},
        });
        let (status, response) = h.post("/v1/responses", body).await;
        assert_eq!(status, StatusCode::OK, "{response}");

        let seen = h.daemon.seen.lock().unwrap()[0].clone();
        assert_eq!(seen.messages.len(), 4, "{:?}", seen.messages);
        let turn = &seen.messages[1];
        assert_eq!(turn.role, Role::Assistant);
        assert_eq!(turn.reasoning.as_deref(), Some("need both"));
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[1].id, "call_2");
        assert_eq!(seen.messages[3].tool_call_id.as_deref(), Some("call_2"));
        assert_eq!(seen.messages[3].parts, vec![Part::Text("15C".into())]);
        assert_eq!(seen.tools[0].name, "get_weather");
        assert_eq!(
            seen.tool_choice,
            Some(ToolChoice::Function("get_weather".into()))
        );
    }

    #[tokio::test]
    async fn a_response_that_calls_tools_lists_the_calls_and_no_empty_message() {
        let mut h = harness(calling(vec![weather_call("call_a", "Paris")]));
        let (_, body) = h.post("/v1/responses", responses_body()).await;
        assert_eq!(body["status"], "completed");
        let output = body["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "{body}");
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["id"], "fc_a");
        assert_eq!(output[0]["call_id"], "call_a");
        assert_eq!(output[0]["arguments"], r#"{"city":"Paris"}"#);
        assert_eq!(output[0]["status"], "completed");
    }

    #[tokio::test]
    async fn a_streamed_call_after_reasoning_follows_the_documented_event_order() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::ReasoningDelta("need it".into())),
            Ok(Event::ToolCall(weather_call("call_a", "Paris"))),
            Ok(Event::Done {
                finish: FinishReason::ToolCalls,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        assert_eq!(
            sse_events(&body),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
            "{body}"
        );
        let data = sse_data(&body);
        let completed = data.last().unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(data[6]["output_index"], 1);
        assert_eq!(data[6]["item"]["arguments"], "");
        assert_eq!(data[7]["delta"], r#"{"city":"Paris"}"#);
    }

    #[tokio::test]
    async fn responses_refuses_tools_it_cannot_serve() {
        responses_refused("tools", json!({"tools": [{"type": "web_search"}]})).await;
        responses_refused("tools", json!({"tools": [{"type": "function"}]})).await;
        responses_refused(
            "tool_choice",
            json!({"tool_choice": {"type": "file_search"}}),
        )
        .await;
        responses_refused(
            "input",
            json!({"input": [{"type": "function_call", "call_id": "c", "name": "f", "arguments": "[]"}]}),
        )
        .await;
        responses_refused(
            "input",
            json!({"input": [{"type": "function_call_output", "output": "21C"}]}),
        )
        .await;
    }

    #[tokio::test]
    async fn a_structured_text_format_is_refused() {
        responses_refused(
            "text.format",
            json!({"text": {"format": {"type": "json_schema"}}}),
        )
        .await;
    }

    #[tokio::test]
    async fn a_plain_text_format_is_allowed() {
        let mut h = harness(Fake::answering("x", None));
        let mut body = responses_body();
        body["text"] = json!({"format": {"type": "text"}});
        let (status, _) = h.post("/v1/responses", body).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn an_image_part_is_refused_until_milestone_three() {
        responses_refused(
            "input",
            json!({"input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA"}]}]}),
        )
        .await;
    }

    // ---- responses, streaming ---------------------------------------------

    /// Event names in order, from the `event:` lines.
    fn sse_events(body: &str) -> Vec<String> {
        body.split("\n\n")
            .filter_map(|block| {
                block
                    .lines()
                    .find(|l| l.starts_with("event:"))
                    .map(|l| l.trim_start_matches("event:").trim().to_string())
            })
            .collect()
    }

    fn streaming_responses_body() -> serde_json::Value {
        let mut body = responses_body();
        body["stream"] = json!(true);
        body
    }

    #[tokio::test]
    async fn a_text_only_stream_follows_the_documented_event_order() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("the sky ".into())),
            Ok(Event::TextDelta("is blue".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ]));
        let (status, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            sse_events(&body),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
            "{body}"
        );
        assert!(!body.contains("[DONE]"), "responses has no DONE sentinel");
    }

    #[tokio::test]
    async fn a_reasoning_stream_brackets_the_reasoning_item_first() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::ReasoningDelta("hmm".into())),
            Ok(Event::TextDelta("blue".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        assert_eq!(
            sse_events(&body),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
            "{body}"
        );
    }

    #[tokio::test]
    async fn sequence_numbers_count_up_from_zero() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("a".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        let numbers: Vec<u64> = sse_data(&body)
            .iter()
            .filter_map(|v| v.get("sequence_number").and_then(|n| n.as_u64()))
            .collect();
        assert!(!numbers.is_empty());
        assert_eq!(
            numbers,
            (0..numbers.len() as u64).collect::<Vec<_>>(),
            "{body}"
        );
    }

    #[tokio::test]
    async fn the_reasoning_item_sits_at_output_index_zero_and_the_message_after_it() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::ReasoningDelta("hmm".into())),
            Ok(Event::TextDelta("blue".into())),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        let data = sse_data(&body);
        let reasoning_delta = data
            .iter()
            .find(|v| v["type"] == "response.reasoning_text.delta")
            .unwrap();
        let text_delta = data
            .iter()
            .find(|v| v["type"] == "response.output_text.delta")
            .unwrap();
        assert_eq!(reasoning_delta["output_index"], 0);
        assert_eq!(text_delta["output_index"], 1);
    }

    #[tokio::test]
    async fn a_budget_cut_off_stream_ends_incomplete() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("cut".into())),
            Ok(Event::Done {
                finish: FinishReason::Length,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        let events = sse_events(&body);
        assert_eq!(events.last().unwrap(), "response.incomplete", "{body}");
        let last = sse_data(&body).pop().unwrap();
        assert_eq!(last["response"]["status"], "incomplete");
        assert_eq!(
            last["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    #[tokio::test]
    async fn a_run_cut_off_while_still_reasoning_still_emits_an_empty_message() {
        // Matches the non-streaming body, which always carries a message item.
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::ReasoningDelta("still thinking".into())),
            Ok(Event::Done {
                finish: FinishReason::Length,
                usage: Usage::default(),
            }),
        ]));
        let (_, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        let last = sse_data(&body).pop().unwrap();
        let output = last["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2, "{body}");
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "");
    }

    #[tokio::test]
    async fn a_failure_partway_through_becomes_response_failed() {
        let mut h = harness(Fake::streaming(vec![
            Ok(Event::TextDelta("the sky ".into())),
            Err(Error::Runtime {
                call: "run_llm".into(),
                code: -1,
            }),
        ]));
        let (status, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            sse_events(&body).last().unwrap(),
            "response.failed",
            "{body}"
        );
        let last = sse_data(&body).pop().unwrap();
        assert_eq!(last["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn the_final_response_carries_the_whole_text_and_usage() {
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
        let (_, body) = h
            .post_raw("/v1/responses", streaming_responses_body())
            .await;
        let last = sse_data(&body).pop().unwrap();
        assert_eq!(last["response"]["status"], "completed");
        assert_eq!(
            last["response"]["output"][0]["content"][0]["text"],
            "the sky is blue"
        );
        assert_eq!(last["response"]["usage"]["total_tokens"], 26);
    }

    // ---- still unimplemented ----------------------------------------------

    #[tokio::test]
    async fn embeddings_still_answers_501() {
        let mut h = harness(Fake::answering("x", None));
        let (status, _) = h.post("/v1/embeddings", json!({})).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    }

    // ---- /v1/audio/transcriptions -----------------------------------------

    /// A daemon that answers a transcription with two segments and a final
    /// event, which is what rkwhisperd's responses map to.
    fn transcribing() -> Fake {
        Fake::streaming(vec![
            Ok(Event::Segment(Segment {
                text: " the sky".into(),
                start_s: 0.0,
                end_s: 1.25,
            })),
            Ok(Event::Segment(Segment {
                text: " is blue".into(),
                start_s: 1.25,
                end_s: 2.0,
            })),
            Ok(Event::Done {
                finish: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ])
    }

    /// A second of 16 kHz mono, which needs no resampling, so a test can check
    /// the bytes that reached the daemon against the ones that went in.
    fn clip() -> Vec<u8> {
        crate::audio::wav(16_000, 1, &crate::audio::sine(16_000, 440.0, 1.0))
    }

    #[tokio::test]
    async fn a_transcription_returns_the_joined_text_as_json() {
        let mut h = harness(transcribing());
        let (status, body, content_type) = h
            .post_form(&[("model", "whisper-small-30s")], Some(("a.wav", &clip())))
            .await;

        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(content_type.as_deref(), Some("application/json"));
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value, json!({"text": "the sky is blue"}));
    }

    #[tokio::test]
    async fn the_decoded_audio_reaches_the_daemon_as_16khz_mono_pcm() {
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(&[("model", "whisper-small-30s")], Some(("a.wav", &clip())))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");

        let (language, pcm) = h.heard();
        assert_eq!(language, None);
        // One second at 16 kHz, two bytes a sample.
        assert_eq!(pcm.len(), 32_000);
    }

    #[tokio::test]
    async fn a_44100_upload_reaches_the_daemon_resampled() {
        let mut h = harness(transcribing());
        let upload = crate::audio::wav(44_100, 1, &crate::audio::sine(44_100, 440.0, 1.0));
        let (status, body, _) = h
            .post_form(&[("model", "whisper-small-30s")], Some(("a.wav", &upload)))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");

        let (_, pcm) = h.heard();
        // A second in, a second out, two bytes a sample.
        assert_eq!(pcm.len(), 32_000);
    }

    #[tokio::test]
    async fn the_language_reaches_the_daemon() {
        let mut h = harness(transcribing());
        h.post_form(
            &[("model", "whisper-small-30s"), ("language", "fr")],
            Some(("a.wav", &clip())),
        )
        .await;
        assert_eq!(h.heard().0, Some("fr".to_string()));
    }

    #[tokio::test]
    async fn an_empty_language_field_is_read_as_unset() {
        let mut h = harness(transcribing());
        h.post_form(
            &[("model", "whisper-small-30s"), ("language", "")],
            Some(("a.wav", &clip())),
        )
        .await;
        assert_eq!(h.heard().0, None);
    }

    #[tokio::test]
    async fn the_text_format_answers_with_the_transcript_alone() {
        let mut h = harness(transcribing());
        let (status, body, content_type) = h
            .post_form(
                &[("model", "whisper-small-30s"), ("response_format", "text")],
                Some(("a.wav", &clip())),
            )
            .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "the sky is blue\n");
        assert!(content_type.unwrap().starts_with("text/plain"));
    }

    #[tokio::test]
    async fn the_srt_format_numbers_its_cues() {
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(
                &[("model", "whisper-small-30s"), ("response_format", "srt")],
                Some(("a.wav", &clip())),
            )
            .await;

        assert_eq!(status, StatusCode::OK);
        assert!(
            body.starts_with("1\n00:00:00,000 --> 00:00:01,250\nthe sky"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn the_vtt_format_opens_with_its_header() {
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(
                &[("model", "whisper-small-30s"), ("response_format", "vtt")],
                Some(("a.wav", &clip())),
            )
            .await;

        assert_eq!(status, StatusCode::OK);
        assert!(
            body.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:01.250\n"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn verbose_json_reports_the_duration_of_the_clip_that_was_sent() {
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(
                &[
                    ("model", "whisper-small-30s"),
                    ("response_format", "verbose_json"),
                ],
                Some(("a.wav", &clip())),
            )
            .await;

        assert_eq!(status, StatusCode::OK, "{body}");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["task"], "transcribe");
        assert_eq!(value["duration"], 1.0);
        assert_eq!(value["segments"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn prompt_and_temperature_are_accepted_and_ignored() {
        // rkwhisper honours neither, and refusing them would break clients that
        // always send them.
        let mut h = harness(transcribing());
        let (status, _, _) = h
            .post_form(
                &[
                    ("model", "whisper-small-30s"),
                    ("prompt", "a hint"),
                    ("temperature", "0.4"),
                ],
                Some(("a.wav", &clip())),
            )
            .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn an_upload_that_is_not_audio_is_refused_before_the_daemon() {
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(
                &[("model", "whisper-small-30s")],
                Some(("a.wav", b"this is not audio at all")),
            )
            .await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["param"], "file");
        assert!(
            h.daemon.heard.lock().unwrap().is_empty(),
            "nothing should have reached the daemon"
        );
    }

    #[tokio::test]
    async fn an_upload_larger_than_axums_own_default_still_goes_through() {
        // Axum's default body limit is 2 MB. This endpoint documents 25, so a
        // 3 MB upload has to reach the decoder rather than being refused while
        // the body is still being read.
        let big = crate::audio::wav(16_000, 1, &crate::audio::sine(16_000, 440.0, 100.0));
        assert!(
            big.len() > 3 * 1024 * 1024,
            "the fixture is {} bytes",
            big.len()
        );

        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(&[("model", "whisper-small-30s")], Some(("a.wav", &big)))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    #[tokio::test]
    async fn an_upload_over_the_limit_is_refused_as_too_large() {
        // Built past MAX_BODY_BYTES, so the limit fires while the body is being
        // read rather than in the size check after it.
        let huge = vec![0u8; crate::audio::MAX_BODY_BYTES + 1];
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(&[("model", "whisper-small-30s")], Some(("a.wav", &huge)))
            .await;

        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("limit"),
            "the message should say it is a size problem, got {body}"
        );
    }

    #[tokio::test]
    async fn a_file_field_over_the_file_limit_is_refused_as_too_large() {
        // Between the two limits: the body is readable, the file is not
        // acceptable. This is the case that names the actual size.
        let over = vec![0u8; crate::audio::MAX_UPLOAD_BYTES + 1];
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(&[("model", "whisper-small-30s")], Some(("a.wav", &over)))
            .await;

        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["param"], "file");
    }

    #[tokio::test]
    async fn a_request_with_no_file_is_refused() {
        let mut h = harness(transcribing());
        let (status, body, _) = h.post_form(&[("model", "whisper-small-30s")], None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["param"], "file");
    }

    #[tokio::test]
    async fn a_request_with_no_model_is_refused() {
        let mut h = harness(transcribing());
        let (status, body, _) = h.post_form(&[], Some(("a.wav", &clip()))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["param"], "model");
    }

    #[tokio::test]
    async fn an_unknown_response_format_is_refused() {
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(
                &[("model", "whisper-small-30s"), ("response_format", "yaml")],
                Some(("a.wav", &clip())),
            )
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["param"], "response_format");
    }

    #[tokio::test]
    async fn asking_for_a_streamed_transcription_is_refused_rather_than_ignored() {
        let mut h = harness(transcribing());
        let (status, body, _) = h
            .post_form(
                &[("model", "whisper-small-30s"), ("stream", "true")],
                Some(("a.wav", &clip())),
            )
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["param"], "stream");
    }

    #[tokio::test]
    async fn a_busy_rkwhisperd_becomes_503_with_retry_after() {
        let mut h = harness(Fake::broken(Error::Busy {
            retry_after_ms: 2_500,
        }));
        let (status, _, _) = h
            .post_form(&[("model", "whisper-small-30s")], Some(("a.wav", &clip())))
            .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn a_transcription_that_never_finishes_is_not_reported_as_complete() {
        // Segments arrived, the final event never did. A partial transcript
        // must not be returned as a whole one.
        let mut h = harness(Fake::streaming(vec![Ok(Event::Segment(Segment {
            text: " the sky".into(),
            start_s: 0.0,
            end_s: 1.25,
        }))]));
        let (status, _, _) = h
            .post_form(&[("model", "whisper-small-30s")], Some(("a.wav", &clip())))
            .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}
