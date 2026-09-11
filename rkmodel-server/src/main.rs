//! The daemon. Loads models onto the NPU and serves them over gRPC.

mod config;
mod generate;
mod registry;
mod service;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use rkmodel_server_protocol::{pb, ModelState, Operation};
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

    // Everything a generate model needs except its weights. A template that
    // does not compile, or names a file that is not there, fails that model
    // and leaves the others alone.
    let mut generate_models = HashMap::new();
    for m in &config.models {
        if !m.operations()?.contains(&Operation::Generate) {
            continue;
        }
        match generate::GenerateModel::load(m) {
            Ok(loaded) => {
                tracing::info!(model = %m.id, "chat template compiled");
                generate_models.insert(m.id.clone(), loaded);
            }
            Err(e) => {
                tracing::error!(model = %m.id, "failed to load: {e:#}");
                registry.set_state(&m.id, ModelState::Failed(format!("{e:#}")));
            }
        }
    }

    for m in registry.list() {
        tracing::info!(model = %m.id, state = m.state.as_str(), "configured");
    }
    tracing::warn!(
        "weights are not loaded yet, so no model reports ready and generate returns unimplemented"
    );

    tracing::info!(%listen, "serving");
    Server::builder()
        .http2_keepalive_interval(Some(Duration::from_secs(10)))
        .http2_keepalive_timeout(Some(Duration::from_secs(5)))
        .tcp_nodelay(true)
        .add_service(pb::rk_model_server_server::RkModelServerServer::new(
            service::Service::new(registry, generate_models),
        ))
        .serve_with_shutdown(listen, async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .context("serving gRPC")?;
    Ok(())
}
