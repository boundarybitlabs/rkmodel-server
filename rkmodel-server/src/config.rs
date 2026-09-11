//! `/etc/rkmodel-server.toml`, the single source of truth for which models
//! exist. The frontend learns the list from `models()` rather than reading this.
//!
//! The whole schema is parsed and validated now so a bad config fails at
//! startup. Several fields have no reader until the model loader lands, which
//! is what the allow below covers; they are deliberately not trimmed, because
//! `deny_unknown_fields` would then reject a config the design already
//! specifies.
#![allow(dead_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rkmodel_server_protocol::Operation;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,

    /// Required when `listen` is not a loopback address.
    #[serde(default)]
    pub token_file: Option<PathBuf>,

    #[serde(default)]
    pub rkwhisper_socket: Option<PathBuf>,

    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

fn default_listen() -> SocketAddr {
    "127.0.0.1:7070".parse().expect("valid default listen addr")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Rkllm,
    Rknn,
    Rkwhisper,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub id: String,
    pub operations: Vec<String>,
    pub backend: Backend,

    // rkllm
    pub rkllm: Option<PathBuf>,
    pub chat_template: Option<PathBuf>,
    pub reasoning: Option<Reasoning>,
    pub max_context_len: Option<u32>,
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    pub sampling: Sampling,
    pub vision: Option<Vision>,

    // rknn
    pub rknn: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub pooling: Option<Pooling>,
    pub normalize: Option<bool>,
    pub core_mask: Option<Vec<u8>>,

    #[serde(default = "default_queue_depth")]
    pub queue_depth: usize,
}

fn default_queue_depth() -> usize {
    8
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reasoning {
    pub start: String,
    pub end: String,
    #[serde(default)]
    pub default: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    Cls,
    Mean,
    Last,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vision {
    pub rknn: PathBuf,
    #[serde(default)]
    pub core_mask: Option<Vec<u8>>,
}

/// All nine, because `InferParams::sampling` takes a whole `Sampling` rather
/// than a partial override. A per-run change to one field needs the other
/// eight, so they are kept here and the request's fields are laid over them.
///
/// Every field defaults, so a model always has a complete set even when the
/// config names none or only some. These values are the daemon's own, not a
/// reading of the runtime's: the runtime resolves its defaults at init and
/// exposes only `n_batch`. The daemon passes this same set to `Param` at load,
/// which is what keeps the two in step.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    #[serde(default = "d_temperature")]
    pub temperature: f32,
    #[serde(default = "d_top_p")]
    pub top_p: f32,
    #[serde(default = "d_top_k")]
    pub top_k: u32,
    #[serde(default = "d_repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub mirostat: u32,
    #[serde(default = "d_mirostat_tau")]
    pub mirostat_tau: f32,
    #[serde(default = "d_mirostat_eta")]
    pub mirostat_eta: f32,
}

fn d_temperature() -> f32 {
    0.8
}
fn d_top_p() -> f32 {
    0.9
}
fn d_top_k() -> u32 {
    40
}
fn d_repeat_penalty() -> f32 {
    1.1
}
fn d_mirostat_tau() -> f32 {
    5.0
}
fn d_mirostat_eta() -> f32 {
    0.1
}

impl Default for Sampling {
    fn default() -> Sampling {
        Sampling {
            temperature: d_temperature(),
            top_p: d_top_p(),
            top_k: d_top_k(),
            repeat_penalty: d_repeat_penalty(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: 0,
            mirostat_tau: d_mirostat_tau(),
            mirostat_eta: d_mirostat_eta(),
        }
    }
}

impl ModelConfig {
    pub fn operations(&self) -> Result<Vec<Operation>> {
        self.operations
            .iter()
            .map(|s| {
                s.parse::<Operation>()
                    .map_err(|_| anyhow::anyhow!("model {}: unknown operation {s:?}", self.id))
            })
            .collect()
    }

    /// Catches a config that names files the backend will not use, or omits
    /// ones it needs, before anything tries to load gigabytes of weights.
    fn validate(&self) -> Result<()> {
        let ops = self.operations()?;
        if ops.is_empty() {
            bail!("model {}: no operations listed", self.id);
        }
        match self.backend {
            Backend::Rkllm => {
                if self.rkllm.is_none() {
                    bail!("model {}: backend rkllm needs an `rkllm` path", self.id);
                }
                if ops.contains(&Operation::Transcribe) {
                    bail!("model {}: backend rkllm cannot transcribe", self.id);
                }
            }
            Backend::Rknn => {
                if self.rknn.is_none() {
                    bail!("model {}: backend rknn needs an `rknn` path", self.id);
                }
                if self.tokenizer.is_none() {
                    bail!("model {}: backend rknn needs a `tokenizer` path", self.id);
                }
                if ops != [Operation::Embed] {
                    bail!("model {}: backend rknn only serves embed", self.id);
                }
            }
            Backend::Rkwhisper => {
                if ops != [Operation::Transcribe] {
                    bail!(
                        "model {}: backend rkwhisper only serves transcribe",
                        self.id
                    );
                }
            }
        }
        Ok(())
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let config: Config =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        // A Unix socket got authentication from file permissions. TCP does not,
        // so listening anywhere reachable requires a token.
        if !self.listen.ip().is_loopback() && self.token_file.is_none() {
            bail!(
                "listen address {} is not loopback, which requires a `token_file`",
                self.listen
            );
        }

        let mut seen = std::collections::HashSet::new();
        for model in &self.models {
            if !seen.insert(model.id.as_str()) {
                bail!("model {} is configured twice", model.id);
            }
            model.validate()?;
        }

        if self.models.iter().any(|m| m.backend == Backend::Rkwhisper)
            && self.rkwhisper_socket.is_none()
        {
            bail!("a rkwhisper model is configured but `rkwhisper_socket` is not set");
        }
        Ok(())
    }

    pub fn token(&self) -> Result<Option<String>> {
        match &self.token_file {
            None => Ok(None),
            Some(path) => {
                let token = std::fs::read_to_string(path)
                    .with_context(|| format!("reading token file {}", path.display()))?;
                let token = token.trim().to_string();
                if token.is_empty() {
                    bail!("token file {} is empty", path.display());
                }
                Ok(Some(token))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<Config> {
        let c: Config = toml::from_str(s)?;
        c.validate()?;
        Ok(c)
    }

    #[test]
    fn loopback_needs_no_token() {
        let c = parse("listen = \"127.0.0.1:7070\"").unwrap();
        assert!(c.token_file.is_none());
    }

    #[test]
    fn non_loopback_requires_a_token_file() {
        let err = parse("listen = \"0.0.0.0:7070\"").unwrap_err().to_string();
        assert!(err.contains("token_file"), "{err}");
    }

    #[test]
    fn rknn_embed_model_parses() {
        let c = parse(
            r#"
            [[models]]
            id = "bge-small-en-v1.5"
            operations = ["embed"]
            backend = "rknn"
            rknn = "/models/bge/model.rknn"
            tokenizer = "/models/bge/tokenizer.json"
            pooling = "cls"
            normalize = true
            core_mask = [0]
            "#,
        )
        .unwrap();
        assert_eq!(c.models[0].pooling, Some(Pooling::Cls));
        assert_eq!(c.models[0].queue_depth, 8);
    }

    #[test]
    fn rkllm_model_needs_its_weights() {
        let err = parse(
            r#"
            [[models]]
            id = "qwen3-4b"
            operations = ["generate"]
            backend = "rkllm"
            "#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("`rkllm` path"), "{err}");
    }

    #[test]
    fn rkwhisper_model_needs_the_socket() {
        let err = parse(
            r#"
            [[models]]
            id = "whisper-small-30s"
            operations = ["transcribe"]
            backend = "rkwhisper"
            "#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("rkwhisper_socket"), "{err}");
    }

    #[test]
    fn a_model_with_no_sampling_block_still_gets_all_nine() {
        let c = parse(
            r#"
            [[models]]
            id = "a"
            operations = ["generate"]
            backend = "rkllm"
            rkllm = "/x.rkllm"
            "#,
        )
        .unwrap();
        assert_eq!(c.models[0].sampling, Sampling::default());
    }

    #[test]
    fn a_partial_sampling_block_defaults_the_rest() {
        let c = parse(
            r#"
            [[models]]
            id = "a"
            operations = ["generate"]
            backend = "rkllm"
            rkllm = "/x.rkllm"
            sampling = { temperature = 0.2 }
            "#,
        )
        .unwrap();
        assert_eq!(c.models[0].sampling.temperature, 0.2);
        assert_eq!(c.models[0].sampling.top_k, Sampling::default().top_k);
    }

    #[test]
    fn duplicate_ids_are_refused() {
        let err = parse(
            r#"
            [[models]]
            id = "a"
            operations = ["generate"]
            backend = "rkllm"
            rkllm = "/x.rkllm"

            [[models]]
            id = "a"
            operations = ["generate"]
            backend = "rkllm"
            rkllm = "/y.rkllm"
            "#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("configured twice"), "{err}");
    }
}
