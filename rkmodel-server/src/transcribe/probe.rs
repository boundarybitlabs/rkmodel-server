//! Whether rkwhisperd is answering.
//!
//! rkwhisper models load nothing in this daemon, so they have no loader thread
//! to report progress the way a generate model does, and the registry would
//! leave them `unavailable` forever. This fills that gap.
//!
//! The check is a session opened and dropped. That is more than reaching the
//! socket, which only says something is listening: the handshake names the
//! model, so it also catches a model rkwhisperd does not serve. It is less than
//! a transcription, since no audio is sent and rkwhisperd does no NPU work.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rkmodel_server_protocol::ModelState;

use super::Asr;
use crate::registry::Registry;

/// How often to re-check. rkwhisperd restarting should be noticed without the
/// probe itself becoming traffic.
///
/// MODEL_SERVER.md leaves "how to tell whether rkwhisperd is up" open, so this
/// is a constant rather than config until that question is settled.
pub const INTERVAL: Duration = Duration::from_secs(30);

/// Reports each model's state into the registry, and keeps reporting.
///
/// One task for all of them: the check is cheap and a handful of whisper models
/// is the whole list.
pub fn spawn(models: Vec<String>, asr: Arc<dyn Asr>, registry: Arc<Registry>) {
    if models.is_empty() {
        return;
    }
    tokio::spawn(async move {
        run(models, asr, registry, INTERVAL).await;
    });
}

async fn run(models: Vec<String>, asr: Arc<dyn Asr>, registry: Arc<Registry>, interval: Duration) {
    // Only transitions are logged. Reporting every model every interval would
    // bury everything else in the journal.
    let mut last: HashMap<String, ModelState> = HashMap::new();

    loop {
        for model in &models {
            let state = match asr.probe(model).await {
                Ok(()) => ModelState::Ready,
                Err(e) => {
                    if last.get(model) != Some(&ModelState::Unavailable) {
                        tracing::warn!(model = %model, "rkwhisperd is not serving this model: {e}");
                    }
                    ModelState::Unavailable
                }
            };

            if last.get(model) != Some(&state) {
                if state.is_ready() {
                    tracing::info!(model = %model, "rkwhisperd is serving this model");
                }
                registry.set_state(model, state.clone());
                last.insert(model.clone(), state);
            }
        }
        tokio::time::sleep(interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::transcribe::Halves;
    use rkmodel_server_protocol::{Error, Operation};
    use std::sync::Mutex;
    use tokio::task::JoinHandle;

    /// Answers the probe from a script, one entry per call.
    ///
    /// The last answer repeats once the script runs out, so the state the loop
    /// settles on is the one the script ends with rather than whatever a race
    /// leaves behind.
    struct FakeAsr {
        answers: Mutex<Vec<Result<(), Error>>>,
    }

    impl FakeAsr {
        fn new(answers: Vec<Result<(), Error>>) -> Arc<FakeAsr> {
            assert!(!answers.is_empty(), "the script needs at least one answer");
            Arc::new(FakeAsr {
                answers: Mutex::new(answers),
            })
        }
    }

    #[async_trait::async_trait]
    impl Asr for FakeAsr {
        async fn open(&self, _: &str, _: Option<&str>) -> Result<Halves, Error> {
            unreachable!("the probe answers from its script, so it opens no session")
        }

        async fn probe(&self, _model: &str) -> Result<(), Error> {
            let mut answers = self.answers.lock().unwrap();
            match answers.len() {
                1 => answers[0].clone(),
                _ => answers.remove(0),
            }
        }
    }

    fn registry() -> Arc<Registry> {
        let config: Config = toml::from_str(
            r#"
            rkwhisper_socket = "/run/rkwhisper/asr.sock"

            [[models]]
            id = "whisper-small-30s"
            operations = ["transcribe"]
            backend = "rkwhisper"
            "#,
        )
        .unwrap();
        Arc::new(Registry::from_config(&config).unwrap())
    }

    /// The probe loop, at an interval short enough for a test. It never returns
    /// on its own, so every caller aborts the handle.
    fn start(asr: Arc<FakeAsr>, registry: Arc<Registry>) -> JoinHandle<()> {
        tokio::spawn(run(
            vec!["whisper-small-30s".to_string()],
            asr,
            registry,
            Duration::from_millis(1),
        ))
    }

    /// Waits for the registry to report `want`.
    ///
    /// Polling the registry rather than counting probe calls: the call happens
    /// before the state is written, so counting calls can read the registry one
    /// round early.
    async fn wait_for(registry: &Registry, want: ModelState) {
        for _ in 0..2_000 {
            if registry.list()[0].state == want {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        panic!(
            "the probe never reported {want:?}; it reports {:?}",
            registry.list()[0].state
        );
    }

    #[test]
    fn a_rkwhisper_model_starts_unavailable() {
        // Nothing loads here, so until the probe answers there is no reason to
        // claim the model is ready.
        let r = registry();
        assert_eq!(r.list()[0].state, ModelState::Unavailable);
        assert!(matches!(
            r.check("whisper-small-30s", Operation::Transcribe),
            Err(Error::Unavailable(_))
        ));
    }

    #[tokio::test]
    async fn a_model_rkwhisperd_serves_becomes_ready() {
        let registry = registry();
        let task = start(FakeAsr::new(vec![Ok(())]), registry.clone());
        wait_for(&registry, ModelState::Ready).await;
        task.abort();

        assert!(registry
            .check("whisper-small-30s", Operation::Transcribe)
            .is_ok());
    }

    #[tokio::test]
    async fn a_ready_model_goes_back_to_unavailable_when_rkwhisperd_stops() {
        let registry = registry();
        let asr = FakeAsr::new(vec![Ok(()), Err(Error::Unreachable("no socket".into()))]);
        let task = start(asr, registry.clone());

        wait_for(&registry, ModelState::Ready).await;
        wait_for(&registry, ModelState::Unavailable).await;
        task.abort();

        assert!(matches!(
            registry.check("whisper-small-30s", Operation::Transcribe),
            Err(Error::Unavailable(_))
        ));
    }

    #[tokio::test]
    async fn a_ready_model_records_when_it_became_available() {
        let registry = registry();
        let task = start(FakeAsr::new(vec![Ok(())]), registry.clone());
        wait_for(&registry, ModelState::Ready).await;
        task.abort();

        assert!(registry.list()[0].loaded_at > 0);
    }

    #[test]
    fn no_models_starts_no_task() {
        // Nothing to probe, and spawning would need a runtime this test has no
        // reason to build.
        spawn(vec![], FakeAsr::new(vec![Ok(())]), registry());
    }
}
