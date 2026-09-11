//! The OpenAI-compatible HTTP frontend. Needs no hardware, and connects to the
//! daemon lazily, so it starts whether or not the daemon is up.

mod error;
mod routes;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use rkmodel_server_client::RkModelClient;

#[derive(Parser)]
#[command(
    name = "rkmodel-server-openai",
    about = "OpenAI-compatible frontend for rkmodel-server."
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

    #[arg(long, default_value = "http://127.0.0.1:7070")]
    daemon: String,

    /// The daemon's shared token, when it requires one.
    #[arg(long)]
    daemon_token_file: Option<PathBuf>,

    /// When set, requests to `/v1/*` must carry `Authorization: Bearer`.
    #[arg(long, env = "RKMODEL_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
}

fn read_trimmed(path: &PathBuf) -> Result<String> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let text = text.trim().to_string();
    anyhow::ensure!(!text.is_empty(), "{} is empty", path.display());
    Ok(text)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let token = args
        .daemon_token_file
        .as_ref()
        .map(read_trimmed)
        .transpose()?;

    let daemon = RkModelClient::new(&args.daemon, token)
        .with_context(|| format!("building a client for {}", args.daemon))?;

    let state = Arc::new(routes::AppState {
        daemon: Arc::new(daemon),
        api_key: args.api_key,
    });
    if state.api_key.is_some() {
        tracing::info!("API key required on /v1");
    }

    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("binding {}", args.listen))?;
    tracing::info!(listen = %args.listen, daemon = %args.daemon, "serving");

    axum::serve(listener, routes::router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .context("serving HTTP")?;
    Ok(())
}
