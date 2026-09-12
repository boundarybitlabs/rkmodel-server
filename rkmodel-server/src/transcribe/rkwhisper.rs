//! [`Asr`] over `rkwhisper-client`, which is the part that needs rkwhisperd.
//!
//! Everything here is a thin wrapper: open the Unix socket, hand over PCM, read
//! responses. The mapping from rkwhisperd's vocabulary to this one lives in the
//! parent module, where it can be tested without a daemon.

use std::path::PathBuf;

use rkmodel_server_protocol::Error;
use rkwhisper_client::asynchronous::{AudioSender, ResponseReceiver, Session};
use rkwhisper_client::{ClientHello, Response};

use super::{Asr, AsrReceiver, AsrSender, Halves, CANCELLED};

/// Opens sessions against rkwhisperd's Unix socket.
pub struct Rkwhisper {
    socket: PathBuf,
}

impl Rkwhisper {
    pub fn new(socket: PathBuf) -> Rkwhisper {
        Rkwhisper { socket }
    }
}

#[async_trait::async_trait]
impl Asr for Rkwhisper {
    async fn open(&self, model: &str, language: Option<&str>) -> Result<Halves, Error> {
        let mut hello = ClientHello {
            model: model.to_string(),
            client_id: "rkmodel-server".to_string(),
            ..ClientHello::default()
        };
        // `ClientHello::default` already carries rkwhisper's default, `en`.
        if let Some(language) = language {
            hello.lang = language.to_string();
        }

        let session = Session::connect(&self.socket, hello).await.map_err(|e| {
            // `connect` retries rkwhisperd's backoff itself, so anything that
            // reaches here is the daemon being unreachable or refusing the
            // model, which the frontend answers 503 for either way.
            tracing::warn!(model, socket = %self.socket.display(), "opening a rkwhisper session failed: {e}");
            map_error(e)
        })?;

        let (sender, receiver) = session.split();
        Ok((Box::new(Sender(sender)), Box::new(Receiver(receiver))))
    }
}

struct Sender(AudioSender);

#[async_trait::async_trait]
impl AsrSender for Sender {
    async fn send_audio(&mut self, pcm: &[u8]) -> Result<(), Error> {
        self.0.send_audio(pcm).await.map_err(map_error)
    }

    async fn finish(&mut self) -> Result<(), Error> {
        self.0.finish().await.map_err(map_error)
    }
}

struct Receiver(ResponseReceiver);

#[async_trait::async_trait]
impl AsrReceiver for Receiver {
    async fn recv(&mut self) -> Result<Response, Error> {
        self.0.recv_response().await.map_err(map_error)
    }
}

/// The client's errors, as this protocol's.
///
/// The client turns rkwhisperd's `Error` response into `Daemon` and its
/// `Cancelled` into `Cancelled` before they get here, which is why those two
/// are mapped in this function rather than beside the other responses.
fn map_error(e: rkwhisper_client::Error) -> Error {
    use rkwhisper_client::Error as Client;
    match e {
        // A closed or missing socket is rkwhisperd being down, not a fault in
        // the request.
        Client::Connection(e) => Error::Unreachable(format!("rkwhisperd: {e}")),
        Client::Handshake(why) => {
            Error::Unavailable(format!("rkwhisperd refused a session: {why}"))
        }
        Client::Daemon(why) => Error::Runtime {
            call: format!("rkwhisperd: {why}"),
            code: -1,
        },
        Client::Cancelled => Error::Runtime {
            call: CANCELLED.into(),
            code: -1,
        },
        Client::Protocol(e) => Error::Runtime {
            call: format!("rkwhisper protocol: {e}"),
            code: -1,
        },
        Client::Other(why) => Error::Runtime {
            call: format!("rkwhisper client: {why}"),
            code: -1,
        },
    }
}
