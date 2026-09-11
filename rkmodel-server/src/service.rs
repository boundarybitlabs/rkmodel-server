//! The gRPC surface. Validation lives here; inference does not.
//!
//! Everything this file checks is settled before a request reaches a model
//! worker, so a bad call costs nothing on the NPU.

use std::pin::Pin;
use std::sync::Arc;

use rkmodel_server_protocol::convert::{operation_from_i32, DecodedInput};
use rkmodel_server_protocol::{pb, Error, Operation, PROTOCOL_VERSION};
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::registry::Registry;

pub struct Service {
    registry: Arc<Registry>,
}

impl Service {
    pub fn new(registry: Arc<Registry>) -> Self {
        Service { registry }
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
        self.registry.check(model, operation)
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
