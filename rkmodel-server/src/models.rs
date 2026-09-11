//! Loading models in the background, and handing the service what it needs.
//!
//! Weights take a long time to read, so a model loads on its own thread. Until
//! it finishes it reports `loading`, and a model that fails to load leaves the
//! others running.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rkmodel_server_protocol::ModelState;

use crate::config::{Backend as ConfigBackend, Config, ModelConfig};
use crate::generate::GenerateModel;
use crate::registry::Registry;
use crate::worker::rkllm::RkllmBackend;
use crate::worker::Worker;

/// A model that is ready to serve `generate`.
pub struct LoadedGenerate {
    /// Template, markers and budgets.
    pub model: GenerateModel,
    /// The thread and queue that own its session.
    pub worker: Worker,
}

#[derive(Default)]
pub struct Models {
    generate: RwLock<HashMap<String, Arc<LoadedGenerate>>>,
}

impl Models {
    pub fn get_generate(&self, id: &str) -> Option<Arc<LoadedGenerate>> {
        self.generate.read().expect("models lock").get(id).cloned()
    }

    pub(crate) fn insert_generate(&self, id: String, loaded: LoadedGenerate) {
        self.generate
            .write()
            .expect("models lock")
            .insert(id, Arc::new(loaded));
    }
}

/// Starts one loader thread per generate model and returns immediately.
///
/// Each thread reports into the registry as it goes, so `GET /health` shows
/// progress rather than blocking on it.
pub fn spawn_loaders(config: &Config, registry: Arc<Registry>, models: Arc<Models>) {
    for model in &config.models {
        let Ok(operations) = model.operations() else {
            continue;
        };
        if !operations.contains(&rkmodel_server_protocol::Operation::Generate) {
            continue;
        }
        if model.backend != ConfigBackend::Rkllm {
            continue;
        }

        let spec = ModelSpec::from(model);
        let registry = registry.clone();
        let models = models.clone();
        std::thread::Builder::new()
            .name(format!("load-{}", model.id))
            .spawn(move || load_generate(spec, registry, models))
            .expect("spawning a loader thread");
    }
}

/// The parts of a model's config a loader thread needs, owned so the thread
/// does not borrow the config.
struct ModelSpec {
    id: String,
    config: ModelConfig,
    queue_depth: usize,
}

impl From<&ModelConfig> for ModelSpec {
    fn from(m: &ModelConfig) -> ModelSpec {
        ModelSpec {
            id: m.id.clone(),
            queue_depth: m.queue_depth,
            config: m.clone(),
        }
    }
}

fn load_generate(spec: ModelSpec, registry: Arc<Registry>, models: Arc<Models>) {
    let id = spec.id.clone();

    // The template first, since it is cheap and a broken one should not cost a
    // multi-gigabyte read before it is noticed.
    let generate = match GenerateModel::load(&spec.config) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!(model = %id, "chat template failed: {e:#}");
            registry.set_state(&id, ModelState::Failed(format!("{e:#}")));
            return;
        }
    };
    tracing::info!(model = %id, "chat template compiled");

    let backend = match RkllmBackend::load(&spec.config) {
        Ok(b) => Arc::new(b),
        Err(e) => {
            tracing::error!(model = %id, "weights failed to load: {e:#}");
            registry.set_state(&id, ModelState::Failed(format!("{e:#}")));
            return;
        }
    };

    let worker = Worker::start(id.clone(), backend, spec.queue_depth);
    models.insert_generate(
        id.clone(),
        LoadedGenerate {
            model: generate,
            worker,
        },
    );
    registry.set_state(&id, ModelState::Ready);
    tracing::info!(model = %id, "ready");
}
