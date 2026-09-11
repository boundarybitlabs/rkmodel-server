//! What the daemon knows about its models.
//!
//! Every model is listed at startup from the config. Loading fills in state
//! later, and a model that fails to load does not stop the others.

use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use rkmodel_server_protocol::{Error, ModelInfo, ModelState, Operation};

use crate::config::{Backend, Config};

pub struct Registry {
    models: RwLock<Vec<ModelInfo>>,
}

// `set_state` and `now` are how the model loader will report progress. They are
// exercised by this module's tests and have no other caller yet.
#[allow(dead_code)]
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Registry {
    pub fn from_config(config: &Config) -> anyhow::Result<Registry> {
        let mut models = Vec::with_capacity(config.models.len());
        for m in &config.models {
            models.push(ModelInfo {
                id: m.id.clone(),
                operations: m.operations()?,
                // Nothing loads weights yet. A model served by rkwhisperd is
                // unavailable until that daemon answers; everything else is
                // still loading.
                state: match m.backend {
                    Backend::Rkwhisper => ModelState::Unavailable,
                    _ => ModelState::Loading,
                },
                loaded_at: 0,
                image_input: None,
                reasoning: m.reasoning.is_some(),
            });
        }
        Ok(Registry {
            models: RwLock::new(models),
        })
    }

    pub fn list(&self) -> Vec<ModelInfo> {
        self.models.read().expect("registry lock").clone()
    }

    #[allow(dead_code)]
    pub fn set_state(&self, id: &str, state: ModelState) {
        let mut models = self.models.write().expect("registry lock");
        if let Some(m) = models.iter_mut().find(|m| m.id == id) {
            if state.is_ready() {
                m.loaded_at = now();
            }
            m.state = state;
        }
    }

    /// The daemon refuses a call when the model is unknown, when the model does
    /// not offer the operation, or when it is not ready to serve.
    pub fn check(&self, model: &str, operation: Operation) -> Result<(), Error> {
        let models = self.models.read().expect("registry lock");
        let info = models
            .iter()
            .find(|m| m.id == model)
            .ok_or_else(|| Error::UnknownModel(model.to_string()))?;

        if !info.offers(operation) {
            return Err(Error::UnsupportedOperation {
                model: model.to_string(),
                operation: operation.to_string(),
            });
        }

        match &info.state {
            ModelState::Ready => Ok(()),
            ModelState::Loading => Err(Error::Loading(model.to_string())),
            ModelState::Unavailable => Err(Error::Unavailable(model.to_string())),
            ModelState::Failed(why) => Err(Error::Runtime {
                call: format!("load {model}: {why}"),
                code: -1,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Registry {
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
        Registry::from_config(&config).unwrap()
    }

    #[test]
    fn unknown_model_is_refused() {
        let r = registry();
        assert!(matches!(
            r.check("nope", Operation::Generate),
            Err(Error::UnknownModel(_))
        ));
    }

    #[test]
    fn operation_the_model_does_not_offer_is_refused() {
        let r = registry();
        r.set_state("qwen3-4b", ModelState::Ready);
        assert!(matches!(
            r.check("qwen3-4b", Operation::Embed),
            Err(Error::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn a_model_still_loading_is_refused() {
        let r = registry();
        assert!(matches!(
            r.check("qwen3-4b", Operation::Generate),
            Err(Error::Loading(_))
        ));
    }

    #[test]
    fn ready_model_passes_and_records_when_it_loaded() {
        let r = registry();
        r.set_state("qwen3-4b", ModelState::Ready);
        assert!(r.check("qwen3-4b", Operation::Generate).is_ok());
        assert!(r.list()[0].loaded_at > 0);
    }
}
