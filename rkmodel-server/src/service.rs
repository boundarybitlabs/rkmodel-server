//! The gRPC surface. Validation lives here; inference does not.
//!
//! Everything this file checks is settled before a request reaches a model
//! worker, so a bad call costs nothing on the NPU.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use rkmodel_server_protocol::convert::{operation_from_i32, DecodedInput};
use rkmodel_server_protocol::{pb, Error, Operation, PROTOCOL_VERSION};
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::generate::GenerateModel;
use crate::registry::Registry;

pub struct Service {
    registry: Arc<Registry>,
    /// Templates and budgets, by model id. The worker reads these once it
    /// exists; the service holds them so a request never waits on a file read.
    generate_models: HashMap<String, GenerateModel>,
}

impl Service {
    pub fn new(registry: Arc<Registry>, generate_models: HashMap<String, GenerateModel>) -> Self {
        Service {
            registry,
            generate_models,
        }
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

        // A generate model reaches ready only with its template compiled. If
        // the two ever disagree, refuse here rather than have the worker reach
        // for a template that is not there.
        if operation == Operation::Generate && !self.generate_models.contains_key(model) {
            return Err(Error::Unavailable(format!(
                "{model} has no chat template loaded"
            )));
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
        for input in req.inputs {
            let decoded = DecodedInput::try_from(input)?;
            self.check_pair(&req.model, operation, &decoded)?;
        }

        Err(Status::unimplemented(format!(
            "{operation}/{} is not wired to a model worker yet",
            req.model
        )))
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

        Err(Status::unimplemented(format!(
            "{operation}/{} is not wired to a model worker yet",
            first.model
        )))
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
    use rkmodel_server_protocol::{GenerateInput, Message, ModelState, Role};

    fn service(with_template: bool) -> Service {
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

        let mut models = HashMap::new();
        if with_template {
            models.insert(
                "qwen3-4b".to_string(),
                crate::generate::GenerateModel {
                    template: crate::generate::template::ChatTemplate::from_source(
                        "{{ messages[0].content }}".into(),
                        Default::default(),
                    )
                    .unwrap(),
                    reasoning: None,
                    sampling: Default::default(),
                    max_context_len: None,
                    max_new_tokens: None,
                },
            );
        }
        Service::new(registry, models)
    }

    fn generate_input() -> DecodedInput {
        DecodedInput::Generate(GenerateInput {
            messages: vec![Message::text(Role::User, "hi")],
            ..Default::default()
        })
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
        let s = service(true);
        let err = s
            .check_pair("qwen3-4b", Operation::Embed, &generate_input())
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn a_matching_pair_on_a_ready_model_passes() {
        let s = service(true);
        assert!(s
            .check_pair("qwen3-4b", Operation::Generate, &generate_input())
            .is_ok());
    }

    #[test]
    fn a_ready_model_with_no_template_is_refused_rather_than_reaching_for_one() {
        let s = service(false);
        let err = s
            .check_pair("qwen3-4b", Operation::Generate, &generate_input())
            .unwrap_err();
        assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    }

    #[test]
    fn an_unknown_model_is_refused_before_the_template_check() {
        let s = service(true);
        let err = s
            .check_pair("nope", Operation::Generate, &generate_input())
            .unwrap_err();
        assert!(matches!(err, Error::UnknownModel(_)), "{err:?}");
    }
}
