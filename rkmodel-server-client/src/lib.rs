//! Talks to rkmodel-server over gRPC.
//!
//! The frontend holds this only as a [`RkModelServer`], so the transport stays
//! behind the trait and tests can swap in a plain implementation.

use std::time::Duration;

use rkmodel_server_protocol::{
    convert, pb, ByteStream, Error, Event, EventStream, Input, ModelInfo, Operation, Output,
    RkModelServer, PROTOCOL_VERSION,
};
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

type Inner = pb::rk_model_server_client::RkModelServerClient<Channel>;

#[derive(Clone)]
pub struct RkModelClient {
    inner: Inner,
    token: Option<String>,
}

impl RkModelClient {
    /// Builds a client without waiting for the daemon. The channel connects
    /// lazily and reconnects on its own, so the frontend can start first and
    /// answer 503 until the daemon is up.
    pub fn new(endpoint: &str, token: Option<String>) -> Result<Self, Error> {
        let channel = Endpoint::from_shared(endpoint.to_string())
            .map_err(|e| Error::InvalidInput(format!("bad daemon endpoint: {e}")))?
            .tcp_nodelay(true)
            .http2_keep_alive_interval(Duration::from_secs(10))
            .keep_alive_timeout(Duration::from_secs(5))
            .keep_alive_while_idle(true)
            .connect_lazy();
        Ok(RkModelClient {
            inner: pb::rk_model_server_client::RkModelServerClient::new(channel),
            token,
        })
    }

    /// Attaches the shared token, when one is configured.
    fn request<T>(&self, body: T) -> Result<Request<T>, Error> {
        let mut req = Request::new(body);
        if let Some(token) = &self.token {
            let value = token
                .parse()
                .map_err(|_| Error::InvalidInput("token is not valid metadata".into()))?;
            req.metadata_mut().insert("authorization", value);
        }
        Ok(req)
    }
}

/// Audio rides as further messages on the same call. Everything else sends one
/// message and then nothing.
fn outbound(
    operation: Operation,
    model: String,
    input: Input,
) -> impl tokio_stream::Stream<Item = pb::StreamRequest> + Send + 'static {
    let (audio, head): (Option<ByteStream>, Input) = match input {
        Input::Transcribe(t) => (
            Some(t.pcm_s16le),
            Input::Transcribe(rkmodel_server_protocol::TranscribeInput {
                pcm_s16le: Box::pin(tokio_stream::empty()),
                language: t.language,
            }),
        ),
        other => (None, other),
    };

    let first = pb::StreamRequest {
        protocol_version: PROTOCOL_VERSION,
        operation: pb::Operation::from(operation) as i32,
        model,
        input: Some(head.into()),
    };

    async_stream::stream! {
        yield first;
        if let Some(mut audio) = audio {
            while let Some(chunk) = audio.next().await {
                match chunk {
                    Ok(bytes) => yield pb::StreamRequest {
                        protocol_version: PROTOCOL_VERSION,
                        operation: pb::Operation::from(operation) as i32,
                        model: String::new(),
                        input: Some(pb::Input {
                            input: Some(pb::input::Input::Transcribe(pb::TranscribeInput {
                                pcm_s16le: bytes,
                                language: None,
                            })),
                        }),
                    },
                    // The frontend failed to decode the upload. Dropping the
                    // outbound stream ends the call, and the daemon cancels.
                    Err(e) => {
                        tracing::warn!("audio stream failed: {e}");
                        break;
                    }
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl RkModelServer for RkModelClient {
    async fn invoke(
        &self,
        operation: Operation,
        model: &str,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, Error> {
        let body = pb::InvokeRequest {
            protocol_version: PROTOCOL_VERSION,
            operation: pb::Operation::from(operation) as i32,
            model: model.to_string(),
            inputs: inputs.into_iter().map(Into::into).collect(),
        };
        let resp = self
            .inner
            .clone()
            .invoke(self.request(body)?)
            .await
            .map_err(Error::from)?;
        resp.into_inner()
            .outputs
            .into_iter()
            .map(Output::try_from)
            .collect()
    }

    async fn invoke_stream(
        &self,
        operation: Operation,
        model: &str,
        input: Input,
    ) -> Result<EventStream, Error> {
        let out = outbound(operation, model.to_string(), input);
        let resp = self
            .inner
            .clone()
            .invoke_stream(self.request(out)?)
            .await
            .map_err(Error::from)?;
        let events = resp
            .into_inner()
            .map(|item: Result<pb::Event, Status>| match item {
                Ok(e) => Event::try_from(e),
                Err(s) => Err(Error::from(s)),
            });
        Ok(Box::pin(events))
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, Error> {
        let resp = self
            .inner
            .clone()
            .models(self.request(pb::ModelsRequest {})?)
            .await
            .map_err(Error::from)?;
        resp.into_inner()
            .models
            .into_iter()
            .map(ModelInfo::try_from)
            .collect()
    }
}

pub use convert::DecodedInput;
