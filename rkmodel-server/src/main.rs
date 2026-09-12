//! The daemon. Loads models onto the NPU and serves them over gRPC.

mod config;
mod generate;
mod models;
mod registry;
mod service;
mod transcribe;
mod worker;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use rkmodel_server_protocol::pb;
use tonic::transport::Server;

#[derive(Parser)]
#[command(name = "rkmodel-server", about = "Serves NPU models over gRPC.")]
struct Args {
    #[arg(long, default_value = "/etc/rkmodel-server.toml")]
    config: PathBuf,

    /// Overrides the config's `listen`.
    #[arg(long)]
    listen: Option<SocketAddr>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let config = config::Config::load(&args.config)
        .with_context(|| format!("loading {}", args.config.display()))?;
    let listen = args.listen.unwrap_or(config.listen);

    // Reading it here surfaces a missing or empty token file at startup rather
    // than on the first call that needs it.
    let token = config.token()?;
    if token.is_some() {
        tracing::info!("token authentication enabled");
    }

    let registry = Arc::new(registry::Registry::from_config(&config)?);
    for m in registry.list() {
        tracing::info!(model = %m.id, state = m.state.as_str(), "configured");
    }

    // Weights take a long time to read, so each model loads on its own thread.
    // The daemon serves straight away, reporting `loading` until they finish,
    // and a model that fails to load leaves the others running.
    let models = Arc::new(models::Models::default());
    models::spawn_loaders(&config, registry.clone(), models.clone());

    // rkwhisperd loads nothing here, so transcribe models have no loader. A
    // probe reports whether that daemon is answering, and keeps reporting.
    let asr = config.rkwhisper_socket.clone().map(|socket| {
        Arc::new(transcribe::rkwhisper::Rkwhisper::new(socket)) as Arc<dyn transcribe::Asr>
    });
    if let Some(asr) = &asr {
        let transcribe_models = config
            .models
            .iter()
            .filter(|m| m.backend == config::Backend::Rkwhisper)
            .map(|m| m.id.clone())
            .collect();
        transcribe::probe::spawn(transcribe_models, asr.clone(), registry.clone());
    }

    tracing::info!(%listen, "serving");
    Server::builder()
        .http2_keepalive_interval(Some(Duration::from_secs(10)))
        .http2_keepalive_timeout(Some(Duration::from_secs(5)))
        .tcp_nodelay(true)
        .add_service(pb::rk_model_server_server::RkModelServerServer::new(
            service::Service::new(registry, models, asr),
        ))
        .serve_with_shutdown(listen, async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .context("serving gRPC")?;
    Ok(())
}
