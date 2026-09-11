//! The daemon's errors, and how they ride a gRPC status.
//!
//! The frontend turns these into HTTP statuses and OpenAI error bodies. The
//! mapping lives in one place so both directions stay in step.

use tonic::{Code, Status};

/// `Busy` carries its backoff here, since gRPC statuses have no field for it.
pub const RETRY_AFTER_MS: &str = "retry-after-ms";

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("unknown model {0}")]
    UnknownModel(String),

    #[error("model {model} does not offer {operation}")]
    UnsupportedOperation { model: String, operation: String },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("context length exceeded: {0}")]
    ContextLengthExceeded(String),

    #[error("busy, retry in {retry_after_ms}ms")]
    Busy { retry_after_ms: u32 },

    #[error("model {0} is still loading")]
    Loading(String),

    #[error("model {0} is unavailable")]
    Unavailable(String),

    #[error("{call} failed with code {code}")]
    Runtime { call: String, code: i32 },

    #[error("unauthorized")]
    Unauthorized,

    #[error("daemon unreachable: {0}")]
    Unreachable(String),

    #[error("protocol version {sent} refused, this daemon speaks {supported}")]
    ProtocolVersion { sent: u32, supported: u32 },
}

impl From<Error> for Status {
    fn from(e: Error) -> Status {
        let code = match &e {
            Error::UnknownModel(_) => Code::NotFound,
            Error::UnsupportedOperation { .. } | Error::InvalidInput(_) => Code::InvalidArgument,
            Error::ContextLengthExceeded(_) => Code::OutOfRange,
            Error::Busy { .. } => Code::ResourceExhausted,
            Error::Loading(_) | Error::Unavailable(_) | Error::Unreachable(_) => Code::Unavailable,
            Error::Runtime { .. } => Code::Internal,
            Error::Unauthorized => Code::Unauthenticated,
            Error::ProtocolVersion { .. } => Code::FailedPrecondition,
        };
        let mut status = Status::new(code, e.to_string());
        if let Error::Busy { retry_after_ms } = e {
            if let Ok(v) = retry_after_ms.to_string().parse() {
                status.metadata_mut().insert(RETRY_AFTER_MS, v);
            }
        }
        status
    }
}

impl From<Status> for Error {
    fn from(s: Status) -> Error {
        let msg = s.message().to_string();
        match s.code() {
            Code::NotFound => Error::UnknownModel(msg),
            Code::InvalidArgument => Error::InvalidInput(msg),
            Code::OutOfRange => Error::ContextLengthExceeded(msg),
            Code::ResourceExhausted => Error::Busy {
                retry_after_ms: s
                    .metadata()
                    .get(RETRY_AFTER_MS)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1000),
            },
            // The transport reports a down daemon with the same code the daemon
            // uses for a model that is not ready, so both land here.
            Code::Unavailable => Error::Unreachable(msg),
            Code::Unauthenticated => Error::Unauthorized,
            Code::FailedPrecondition => Error::ProtocolVersion {
                sent: 0,
                supported: crate::PROTOCOL_VERSION,
            },
            _ => Error::Runtime {
                call: "invoke".into(),
                code: s.code() as i32,
            },
        }
    }
}

impl From<tonic::transport::Error> for Error {
    fn from(e: tonic::transport::Error) -> Error {
        Error::Unreachable(e.to_string())
    }
}

/// Everything the frontend needs to answer a failed request.
pub struct HttpMapping {
    pub status: u16,
    /// OpenAI's `error.type`.
    pub kind: &'static str,
    /// OpenAI's `error.code`, when there is a specific one.
    pub code: Option<&'static str>,
    pub retry_after_ms: Option<u32>,
}

impl Error {
    pub fn http(&self) -> HttpMapping {
        let invalid = "invalid_request_error";
        let server = "server_error";
        match self {
            Error::UnknownModel(_) => HttpMapping {
                status: 404,
                kind: invalid,
                code: Some("model_not_found"),
                retry_after_ms: None,
            },
            Error::UnsupportedOperation { .. } | Error::InvalidInput(_) => HttpMapping {
                status: 400,
                kind: invalid,
                code: None,
                retry_after_ms: None,
            },
            Error::ContextLengthExceeded(_) => HttpMapping {
                status: 400,
                kind: invalid,
                code: Some("context_length_exceeded"),
                retry_after_ms: None,
            },
            Error::Busy { retry_after_ms } => HttpMapping {
                status: 503,
                kind: server,
                code: None,
                retry_after_ms: Some(*retry_after_ms),
            },
            Error::Loading(_) | Error::Unavailable(_) | Error::Unreachable(_) => HttpMapping {
                status: 503,
                kind: server,
                code: None,
                retry_after_ms: None,
            },
            // A refused token or protocol version means the frontend is
            // misconfigured, which is a server fault, not the caller's.
            Error::Runtime { .. } | Error::Unauthorized | Error::ProtocolVersion { .. } => {
                HttpMapping {
                    status: 500,
                    kind: server,
                    code: None,
                    retry_after_ms: None,
                }
            }
        }
    }
}
