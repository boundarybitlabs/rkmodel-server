//! The gRPC surface. Validation lives here; inference does not.
//!
//! Everything this file checks is settled before a request reaches a model
//! worker, so a bad call costs nothing on the NPU.

use std::pin::Pin;
use std::sync::Arc;

use rkmodel_server_protocol::convert::{operation_from_i32, DecodedInput};
use rkmodel_server_protocol::{
    pb, Error, Event, GenerateInput, Operation, Output, PROTOCOL_VERSION,
};
use tokio::sync::mpsc;

use crate::worker::RunHandle;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::generate::sampling;
use crate::models::{LoadedGenerate, Models};
use crate::registry::Registry;

pub struct Service {
    registry: Arc<Registry>,
    models: Arc<Models>,
}

impl Service {
    pub fn new(registry: Arc<Registry>, models: Arc<Models>) -> Self {
        Service { registry, models }
    }

    /// Everything between a validated request and a queued run: render the
    /// transcript with the model's own template, resolve sampling and the
    /// budget, and build the reasoning splitter for this run.
    fn queue_generate(
        &self,
        model: &str,
        input: GenerateInput,
    ) -> Result<(mpsc::Receiver<Result<Event, Error>>, RunHandle), Error> {
        let loaded: Arc<LoadedGenerate> = self
            .models
            .get_generate(model)
            .ok_or_else(|| Error::Unavailable(format!("{model} is not loaded")))?;

        let reasoning_on = loaded.model.wants_reasoning(input.reasoning);
        let prompt = loaded
            .model
            .template
            .render(&input.messages, reasoning_on)
            .map_err(|e| Error::InvalidInput(format!("rendering the prompt failed: {e:#}")))?;

        let parser = loaded.model.parser(&prompt, reasoning_on);
        let budgets = sampling::resolve(loaded.model.sampling, loaded.model.max_new_tokens, &input);

        loaded
            .worker
            .submit(prompt, budgets.sampling, budgets.max_new_tokens, parser)
    }

    /// Runs one generation to completion and folds its events into one output.
    async fn generate_once(&self, model: &str, input: GenerateInput) -> Result<Output, Error> {
        let (mut rx, handle) = self.queue_generate(model, input)?;

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut done = None;

        while let Some(event) = rx.recv().await {
            match event? {
                Event::TextDelta(s) => text.push_str(&s),
                Event::ReasoningDelta(s) => reasoning.push_str(&s),
                Event::Done { finish, usage } => done = Some((finish, usage)),
                Event::Segment(_) => {}
            }
        }
        drop(handle);

        let (finish, usage) = done.ok_or_else(|| Error::Runtime {
            call: format!("generate on {model} ended without a final event"),
            code: -1,
        })?;

        Ok(Output::Generated {
            text,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            finish,
            usage,
        })
    }

    fn check_version(sent: u32) -> Result<(), Error> {
        if sent == PROTOCOL_VERSION {
            Ok(())
        } else {
            Err(Error::ProtocolVersion {
                sent,
                supported: PROTOCOL_VERSION,
            })
        }
    }

    /// The pair has to agree with itself before it is checked against config.
    fn check_pair(
        &self,
        model: &str,
        operation: Operation,
        input: &DecodedInput,
    ) -> Result<(), Error> {
        if input.operation() != operation {
            return Err(Error::InvalidInput(format!(
                "input is a {} input, but the call names {operation}",
                input.operation()
            )));
        }
        self.registry.check(model, operation)?;

        // A generate model reaches ready only once its worker exists. If the
        // registry and the loaded set ever disagree, refuse here rather than
        // have the request reach for a worker that is not there.
        if operation == Operation::Generate && self.models.get_generate(model).is_none() {
            return Err(Error::Unavailable(format!("{model} is not loaded")));
        }
        Ok(())
    }
}

type EventStream = Pin<Box<dyn Stream<Item = Result<pb::Event, Status>> + Send>>;

#[tonic::async_trait]
impl pb::rk_model_server_server::RkModelServer for Service {
    async fn invoke(
        &self,
        request: Request<pb::InvokeRequest>,
    ) -> Result<Response<pb::InvokeResponse>, Status> {
        let req = request.into_inner();
        Service::check_version(req.protocol_version)?;
        let operation = operation_from_i32(req.operation)?;

        if req.inputs.is_empty() {
            return Err(Error::InvalidInput("no inputs sent".into()).into());
        }
        // All-or-nothing: one bad input fails the call, and nothing runs.
        let mut decoded = Vec::with_capacity(req.inputs.len());
        for input in req.inputs {
            let one = DecodedInput::try_from(input)?;
            self.check_pair(&req.model, operation, &one)?;
            decoded.push(one);
        }

        let mut outputs = Vec::with_capacity(decoded.len());
        for one in decoded {
            let output = match one {
                DecodedInput::Generate(g) => self.generate_once(&req.model, g).await?,
                other => {
                    return Err(Status::unimplemented(format!(
                        "{} is not wired to a model worker yet",
                        other.operation()
                    )))
                }
            };
            outputs.push(output.into());
        }
        Ok(Response::new(pb::InvokeResponse { outputs }))
    }

    type InvokeStreamStream = EventStream;

    async fn invoke_stream(
        &self,
        request: Request<Streaming<pb::StreamRequest>>,
    ) -> Result<Response<Self::InvokeStreamStream>, Status> {
        let mut inbound = request.into_inner();

        // The first message carries the request. For transcribe, audio chunks
        // follow on the same call.
        let first = inbound
            .next()
            .await
            .transpose()?
            .ok_or_else(|| Status::from(Error::InvalidInput("no request sent".into())))?;

        Service::check_version(first.protocol_version)?;
        let operation = operation_from_i32(first.operation)?;
        let input = first
            .input
            .ok_or_else(|| Status::from(Error::InvalidInput("no input sent".into())))?;
        let decoded = DecodedInput::try_from(input)?;
        self.check_pair(&first.model, operation, &decoded)?;

        let DecodedInput::Generate(generate) = decoded else {
            return Err(Status::unimplemented(format!(
                "{operation} streaming is not wired to a model worker yet"
            )));
        };

        let (mut rx, handle) = self.queue_generate(&first.model, generate)?;
        let events = async_stream::stream! {
            // Dropped with the stream, which is what tonic does when the peer
            // sends RST_STREAM. That cancels the run.
            let _handle = handle;
            while let Some(event) = rx.recv().await {
                match event {
                    Ok(e) => yield Ok(pb::Event::from(e)),
                    Err(e) => {
                        yield Err(Status::from(e));
                        return;
                    }
                }
            }
        };
        Ok(Response::new(Box::pin(events)))
    }

    async fn models(
        &self,
        _request: Request<pb::ModelsRequest>,
    ) -> Result<Response<pb::ModelsResponse>, Status> {
        Ok(Response::new(pb::ModelsResponse {
            models: self.registry.list().into_iter().map(Into::into).collect(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::generate::template::ChatTemplate;
    use crate::generate::GenerateModel;
    use crate::models::LoadedGenerate;
    use crate::worker::backend::fake::FakeBackend;
    use crate::worker::Worker;
    use rkmodel_server_protocol::{Message, ModelState, Role};

    fn service(loaded: bool) -> Service {
        service_with_seen(loaded).0
    }

    fn service_with_seen(loaded: bool) -> (Service, Option<Arc<std::sync::Mutex<Vec<String>>>>) {
        let config: Config = toml::from_str(
            r#"
            [[models]]
            id = "qwen3-4b"
            operations = ["generate"]
            backend = "rkllm"
            rkllm = "/models/qwen3-4b.rkllm"
            "#,
        )
        .unwrap();
        let registry = Arc::new(Registry::from_config(&config).unwrap());
        registry.set_state("qwen3-4b", ModelState::Ready);

        let models = Arc::new(Models::default());
        let mut seen = None;
        if loaded {
            let template = ChatTemplate::from_source(
                "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant:"
                    .into(),
                Default::default(),
            )
            .unwrap();
            models.insert_generate(
                "qwen3-4b".to_string(),
                LoadedGenerate {
                    model: GenerateModel {
                        template,
                        reasoning: None,
                        sampling: Default::default(),
                        max_context_len: None,
                        max_new_tokens: None,
                    },
                    worker: {
                        let backend = Arc::new(FakeBackend::new(&["the sky ", "is blue"]));
                        seen = Some(backend.seen.clone());
                        Worker::start("qwen3-4b".into(), backend, 4)
                    },
                },
            );
        }
        (Service::new(registry, models), seen)
    }

    fn generate_input() -> GenerateInput {
        GenerateInput {
            messages: vec![Message::text(Role::User, "why is the sky blue?")],
            ..Default::default()
        }
    }

    fn decoded() -> DecodedInput {
        DecodedInput::Generate(generate_input())
    }

    #[test]
    fn the_supported_protocol_version_passes() {
        assert!(Service::check_version(1).is_ok());
    }

    #[test]
    fn another_protocol_version_is_refused_naming_both() {
        let err = Service::check_version(99).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("99") && message.contains('1'), "{message}");
    }

    #[test]
    fn an_input_that_does_not_match_the_operation_is_refused() {
        let err = service(true)
            .check_pair("qwen3-4b", Operation::Embed, &decoded())
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn a_matching_pair_on_a_loaded_model_passes() {
        assert!(service(true)
            .check_pair("qwen3-4b", Operation::Generate, &decoded())
            .is_ok());
    }

    #[test]
    fn a_ready_model_that_is_not_loaded_is_refused() {
        let err = service(false)
            .check_pair("qwen3-4b", Operation::Generate, &decoded())
            .unwrap_err();
        assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    }

    #[test]
    fn an_unknown_model_is_refused_before_the_loaded_check() {
        let err = service(true)
            .check_pair("nope", Operation::Generate, &decoded())
            .unwrap_err();
        assert!(matches!(err, Error::UnknownModel(_)), "{err:?}");
    }

    #[tokio::test]
    async fn a_generate_call_renders_the_prompt_and_returns_one_output() {
        let s = service(true);
        let output = s.generate_once("qwen3-4b", generate_input()).await.unwrap();
        match output {
            Output::Generated {
                text,
                reasoning,
                finish,
                usage,
            } => {
                assert_eq!(text, "the sky is blue");
                assert_eq!(reasoning, None);
                assert_eq!(finish, rkmodel_server_protocol::FinishReason::Stop);
                assert_eq!(usage.output_tokens, 2);
            }
            other => panic!("expected Generated, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_backend_sees_the_prompt_the_template_rendered() {
        let (s, seen) = service_with_seen(true);
        s.generate_once("qwen3-4b", generate_input()).await.unwrap();
        let seen = seen.unwrap();
        let prompts = seen.lock().unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], "user: why is the sky blue?\nassistant:");
    }

    #[tokio::test]
    async fn a_model_that_is_not_loaded_cannot_be_queued() {
        let err = match service(false).queue_generate("qwen3-4b", generate_input()) {
            Err(e) => e,
            Ok(_) => panic!("an unloaded model should not queue"),
        };
        assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    }
}
