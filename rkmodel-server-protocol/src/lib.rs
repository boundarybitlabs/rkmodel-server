//! The wire protocol, and the vocabulary the rest of the workspace speaks.
//!
//! [`pb`] is the `tonic-build` output, and nothing outside this crate and
//! `rkmodel-server-client` should need it. Everything else works in the types
//! in [`types`], which carry no proto details.

pub mod convert;
pub mod error;
pub mod types;

/// Generated from `proto/rkmodel.proto`.
pub mod pb {
    // tonic's generated methods return `Status` in the error position, which
    // clippy reads as a large `Err`. Nothing here is hand-written.
    #![allow(clippy::result_large_err)]

    tonic::include_proto!("rkmodel.v1");
}

pub use error::Error;
pub use types::*;

/// The protocol version carried on every request. The major version also sits
/// in the proto package name, so a mismatch there fails at the route instead.
///
/// 2 added tools. A frontend at 1 would drop a tool call it cannot decode, so
/// the daemon refuses it rather than let tool calls vanish.
pub const PROTOCOL_VERSION: u32 = 2;

use std::pin::Pin;

/// A stream of generation events. The design sketch writes this as
/// `impl Stream`, but boxing keeps the trait usable behind `dyn`, which the
/// frontend's fake daemon relies on in tests.
pub type EventStream = Pin<Box<dyn futures_core::Stream<Item = Result<Event, Error>> + Send>>;

/// Chunks of 16 kHz mono s16le audio, as the frontend decodes them.
pub type ByteStream = Pin<Box<dyn futures_core::Stream<Item = Result<Vec<u8>, Error>> + Send>>;

/// The daemon, as its clients see it.
///
/// `rkmodel-server-client` implements this over tonic. The frontend holds
/// nothing else, so the transport stays swappable and tests can substitute a
/// plain in-process implementation.
#[async_trait::async_trait]
pub trait RkModelServer: Send + Sync {
    /// One output per input, in the same order. All-or-nothing: one bad input
    /// fails the call.
    async fn invoke(
        &self,
        operation: Operation,
        model: &str,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, Error>;

    /// One input, streamed. Ends with [`Event::Done`] or an error.
    async fn invoke_stream(
        &self,
        operation: Operation,
        model: &str,
        input: Input,
    ) -> Result<EventStream, Error>;

    /// Every configured model, with its operations and state.
    async fn models(&self) -> Result<Vec<ModelInfo>, Error>;
}
