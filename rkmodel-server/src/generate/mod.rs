//! Turning a request into a prompt, and model output back into events.
//!
//! None of this touches the NPU, so all of it is tested on any host.
//!
//! Startup already uses this: every generate model's template is compiled here,
//! and a model whose template is missing or broken is marked failed rather than
//! erroring on its first request. The reasoning splitter and the sampling
//! overlay have only one future caller, the model worker, so they read as dead
//! code until it lands. They are covered by this module's tests meanwhile.
#![allow(dead_code)]

pub mod output;
pub mod reasoning;
pub mod sampling;
pub mod template;
pub mod tools;

use std::path::Path;

use anyhow::{Context, Result};
use rkmodel_server_protocol::{Error, GenerateInput, ToolChoice};
use tokenizers::Tokenizer;

use crate::config::{ModelConfig, Sampling, ToolFormat};
use crate::worker::Prompt;
use template::ChatTemplate;
use tools::{ForcedStart, ToolCallParser};

/// Everything a `generate` model needs that is not the weights.
///
/// Built at startup so a broken template is a load failure with a reason,
/// rather than an error on the first request.
pub struct GenerateModel {
    pub template: ChatTemplate,
    /// The markers around this model's reasoning, when it reasons.
    pub reasoning: Option<ReasoningMarkers>,
    pub sampling: Sampling,
    pub max_context_len: Option<u32>,
    pub max_new_tokens: Option<u32>,
    /// Set for a model the daemon tokenizes itself, which is Plan B.
    ///
    /// Measured on the board, the runtime's own tokenizer split a Gemma 4 E2B
    /// tool prompt into 99 tokens where Hugging Face's makes 92, and Gemma only
    /// called the tool when given the 92.
    pub tokenizer: Option<Tokenizer>,
    /// How this model writes a tool call. `None` refuses tools.
    pub tool_format: Option<ToolFormat>,
}

/// How a request's tools shape its run.
pub struct ToolRun {
    /// Whether the template is given the tools. `tool_choice: none` leaves
    /// them out, though earlier tool turns still render.
    pub offered: bool,
    /// The start of a call written into the prompt, for a choice that forces
    /// one.
    pub forced: Option<ForcedStart>,
    /// `None` when calls are not looked for.
    pub parser: Option<ToolCallParser>,
}

impl ToolRun {
    fn none() -> ToolRun {
        ToolRun {
            offered: false,
            forced: None,
            parser: None,
        }
    }
}

/// OpenAI's rule for a function name, which also keeps a name safe to write
/// into a prompt when a choice forces it.
fn valid_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

#[derive(Debug, Clone)]
pub struct ReasoningMarkers {
    pub start: String,
    pub end: String,
    /// What a request that says nothing about reasoning gets.
    pub default_on: bool,
}

impl GenerateModel {
    /// Reads the template, and the special tokens that must not survive in user
    /// text. Does not load weights.
    pub fn load(model: &ModelConfig) -> Result<GenerateModel> {
        let path = model
            .chat_template
            .as_deref()
            .with_context(|| format!("model {} has no chat_template", model.id))?;

        let mut template = ChatTemplate::load(path)?;

        // A template shipped as bare Jinja carries no token list, so the
        // special tokens come from the model's `tokenizer.json`: the configured
        // one, or one beside the template.
        let tokenizer_path = model.tokenizer.clone().or_else(|| sibling_tokenizer(path));
        if let Some(tokenizer) = &tokenizer_path {
            let extra = template::special_from_tokenizer_json(tokenizer)
                .with_context(|| format!("reading special tokens for model {}", model.id))?;
            template.add_special_tokens(extra);
        }

        if let Some(format) = model.tool_format {
            if !template.reads_tools() {
                anyhow::bail!(
                    "model {} has a tool_format, but its chat template never reads `tools`",
                    model.id
                );
            }
            template.add_special_tokens(format.reserved().iter().map(|m| m.to_string()));
        }

        // Only a configured tokenizer switches to Plan B. One that merely sits
        // beside the template is read for its special tokens and nothing else.
        let tokenizer = match &model.tokenizer {
            Some(path) => Some(Tokenizer::from_file(path).map_err(|e| {
                anyhow::anyhow!(
                    "loading tokenizer {} for model {}: {e}",
                    path.display(),
                    model.id
                )
            })?),
            None => None,
        };

        Ok(GenerateModel {
            template,
            reasoning: model.reasoning.as_ref().map(|r| ReasoningMarkers {
                start: r.start.clone(),
                end: r.end.clone(),
                default_on: r.default,
            }),
            sampling: model.sampling,
            max_context_len: model.max_context_len,
            max_new_tokens: model.max_new_tokens,
            tokenizer,
            tool_format: model.tool_format,
        })
    }

    /// Checks a request's tools against this model, and works out what they
    /// ask of the run.
    pub fn tool_run(&self, input: &GenerateInput) -> Result<ToolRun, Error> {
        let choice = input.tool_choice.clone().unwrap_or(ToolChoice::Auto);
        if input.tools.is_empty() {
            return match choice {
                ToolChoice::Required | ToolChoice::Function(_) => Err(Error::InvalidInput(
                    "tool_choice asks for a call, but no tools were given".into(),
                )),
                ToolChoice::None | ToolChoice::Auto => Ok(ToolRun::none()),
            };
        }
        let Some(format) = self.tool_format else {
            return Err(Error::InvalidInput("this model does not take tools".into()));
        };

        let mut names: Vec<String> = Vec::with_capacity(input.tools.len());
        for tool in &input.tools {
            if !valid_tool_name(&tool.name) {
                return Err(Error::InvalidInput(format!(
                    "tool name {:?} must be 1 to 64 letters, digits, underscores or dashes",
                    tool.name
                )));
            }
            if names.contains(&tool.name) {
                return Err(Error::InvalidInput(format!(
                    "tool {} is declared twice",
                    tool.name
                )));
            }
            names.push(tool.name.clone());
        }
        if let ToolChoice::Function(name) = &choice {
            if !names.contains(name) {
                return Err(Error::InvalidInput(format!(
                    "tool_choice names {name}, which is not among the tools"
                )));
            }
        }
        if choice == ToolChoice::None {
            return Ok(ToolRun::none());
        }

        let forced = format.forced_start(&choice);
        let stop_after_first = input.parallel_tool_calls == Some(false);
        let mut parser = ToolCallParser::new(format, names, stop_after_first);
        if let Some(start) = &forced {
            parser = parser.primed(&start.body);
        }
        Ok(ToolRun {
            offered: true,
            forced,
            parser: Some(parser),
        })
    }

    /// What the backend runs for a rendered prompt: the text alone under
    /// Plan A, or with its token ids under Plan B.
    ///
    /// Only Plan B knows the prompt's length before the run, so only it refuses
    /// a prompt that does not fit, rather than leaving that to the runtime.
    pub fn prompt(&self, text: String, max_new_tokens: Option<u32>) -> Result<Prompt, Error> {
        let Some(tokenizer) = &self.tokenizer else {
            return Ok(Prompt::from(text));
        };
        // No special tokens added: the template writes `bos_token` itself, and
        // adding another would give the model two.
        let encoding = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| Error::InvalidInput(format!("tokenizing the prompt failed: {e}")))?;
        let tokens: Vec<i32> = encoding
            .get_ids()
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| {
                    Error::InvalidInput(format!("token id {id} does not fit the runtime's i32"))
                })
            })
            .collect::<Result<_, _>>()?;

        let length = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        sampling::check_context_length(length, self.max_context_len, max_new_tokens)?;

        Ok(Prompt {
            text,
            tokens: Some(tokens),
        })
    }

    /// Whether this run should reason. `None` from the request takes the
    /// model's configured default.
    pub fn wants_reasoning(&self, requested: Option<bool>) -> bool {
        match (&self.reasoning, requested) {
            (None, _) => false,
            (Some(markers), None) => markers.default_on,
            (Some(_), Some(on)) => on,
        }
    }

    /// Builds the splitter for one run, told whether the rendered prompt
    /// already opened the reasoning block.
    pub fn parser(&self, prompt: &str, reasoning_on: bool) -> reasoning::ReasoningParser {
        match &self.reasoning {
            Some(m) if reasoning_on => {
                let opened = reasoning::ReasoningParser::prompt_opens_reasoning(prompt, &m.start);
                reasoning::ReasoningParser::new(&m.start, &m.end, opened)
            }
            // With reasoning off the model should emit none, but a model that
            // emits it anyway still gets split rather than leaking markers.
            Some(m) => reasoning::ReasoningParser::new(&m.start, &m.end, false),
            None => reasoning::ReasoningParser::disabled(),
        }
    }
}

/// `tokenizer.json` next to the template, when the model ships one.
fn sibling_tokenizer(template: &Path) -> Option<std::path::PathBuf> {
    let candidate = template.parent()?.join("tokenizer.json");
    candidate.exists().then_some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Reasoning;

    fn model_config(chat_template: Option<&str>, reasoning: Option<Reasoning>) -> ModelConfig {
        let mut c: ModelConfig = toml::from_str(
            r#"
            id = "qwen3-4b"
            operations = ["generate"]
            backend = "rkllm"
            rkllm = "/models/qwen3-4b.rkllm"
            "#,
        )
        .unwrap();
        c.chat_template = chat_template.map(Into::into);
        c.reasoning = reasoning;
        c
    }

    /// `GenerateModel` holds a compiled template, which has no `Debug`, so
    /// `unwrap_err` is not available.
    fn load_err(config: &ModelConfig) -> anyhow::Error {
        match GenerateModel::load(config) {
            Err(e) => e,
            Ok(_) => panic!("expected this model to fail to load"),
        }
    }

    fn markers() -> Reasoning {
        toml::from_str(
            r#"start = "<think>"
end = "</think>"
default = false"#,
        )
        .unwrap()
    }

    /// A word-level tokenizer with `<bos>` as an added special token, and a
    /// post-processor that would prepend a second `<bos>` if asked to add
    /// special tokens.
    fn tiny_tokenizer() -> Tokenizer {
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{"id": 0, "content": "<bos>", "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}},
                           {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                         {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {"<bos>": {"id": "<bos>", "ids": [0], "tokens": ["<bos>"]}}},
            "decoder": null,
            "model": {"type": "WordLevel",
                "vocab": {"<bos>": 0, "<unk>": 1, "hello": 2, "world": 3},
                "unk_token": "<unk>"}
        }"#
        .parse()
        .unwrap()
    }

    fn plan_b(max_context_len: Option<u32>) -> GenerateModel {
        GenerateModel {
            template: ChatTemplate::from_source("x".into(), Default::default()).unwrap(),
            reasoning: None,
            sampling: Sampling::default(),
            max_context_len,
            max_new_tokens: None,
            tokenizer: Some(tiny_tokenizer()),
            tool_format: None,
        }
    }

    fn with_tools() -> GenerateModel {
        let mut m = plan_b(None);
        m.tokenizer = None;
        m.tool_format = Some(ToolFormat::Hermes);
        m
    }

    fn tools_input(names: &[&str], choice: Option<ToolChoice>) -> GenerateInput {
        GenerateInput {
            tools: names
                .iter()
                .map(|n| rkmodel_server_protocol::Tool {
                    name: n.to_string(),
                    description: None,
                    parameters_json: None,
                })
                .collect(),
            tool_choice: choice,
            ..Default::default()
        }
    }

    fn invalid(result: Result<ToolRun, Error>) -> String {
        match result {
            Err(Error::InvalidInput(m)) => m,
            Err(other) => panic!("expected InvalidInput, got {other:?}"),
            Ok(_) => panic!("expected InvalidInput, got a run"),
        }
    }

    #[test]
    fn tool_choices_decide_what_the_run_does() {
        let m = with_tools();
        let auto = m.tool_run(&tools_input(&["f"], None)).unwrap();
        assert!(auto.offered && auto.forced.is_none() && auto.parser.is_some());

        let none = m
            .tool_run(&tools_input(&["f"], Some(ToolChoice::None)))
            .unwrap();
        assert!(!none.offered && none.parser.is_none());

        let forced = m
            .tool_run(&tools_input(
                &["f", "g"],
                Some(ToolChoice::Function("g".into())),
            ))
            .unwrap();
        assert_eq!(
            forced.forced.unwrap().prompt,
            "<tool_call>\n{\"name\": \"g\", \"arguments\": "
        );

        let nothing = m.tool_run(&tools_input(&[], None)).unwrap();
        assert!(!nothing.offered && nothing.parser.is_none());
    }

    #[test]
    fn bad_tool_requests_are_refused_with_a_reason() {
        let m = with_tools();
        let msg = invalid(m.tool_run(&tools_input(&["f"], Some(ToolChoice::Function("g".into())))));
        assert!(msg.contains("not among the tools"), "{msg}");

        let msg = invalid(m.tool_run(&tools_input(&["f\"}; evil"], None)));
        assert!(msg.contains("letters, digits"), "{msg}");

        let msg = invalid(m.tool_run(&tools_input(&["f", "f"], None)));
        assert!(msg.contains("twice"), "{msg}");

        let msg = invalid(m.tool_run(&tools_input(&[], Some(ToolChoice::Required))));
        assert!(msg.contains("no tools"), "{msg}");

        let mut plain = with_tools();
        plain.tool_format = None;
        let msg = invalid(plain.tool_run(&tools_input(&["f"], None)));
        assert!(msg.contains("does not take tools"), "{msg}");
    }

    #[test]
    fn a_tool_format_with_a_template_that_ignores_tools_fails_to_load() {
        let dir = std::env::temp_dir().join(format!("rkmodel-tools-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let template = dir.join("chat_template.jinja");
        std::fs::write(&template, "{{ messages[0].content }}").unwrap();

        let mut config = model_config(template.to_str(), None);
        config.tool_format = Some(ToolFormat::Hermes);
        let err = load_err(&config);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            format!("{err:#}").contains("never reads `tools`"),
            "{err:#}"
        );
    }

    #[test]
    fn plan_a_hands_over_the_text_alone() {
        let mut m = plan_b(None);
        m.tokenizer = None;
        let prompt = m.prompt("<bos>hello".into(), None).unwrap();
        assert_eq!(prompt, Prompt::from("<bos>hello"));
    }

    #[test]
    fn plan_b_tokenizes_the_rendered_bos_once() {
        // The template wrote `<bos>`. The tokenizer's own post-processor would
        // add another, which is why special tokens are not added.
        let prompt = plan_b(None)
            .prompt("<bos>hello world".into(), None)
            .unwrap();
        assert_eq!(prompt.text, "<bos>hello world");
        assert_eq!(prompt.tokens, Some(vec![0, 2, 3]));
    }

    #[test]
    fn plan_b_refuses_a_prompt_that_does_not_fit() {
        let err = plan_b(Some(3))
            .prompt("<bos>hello world".into(), None)
            .unwrap_err();
        assert!(matches!(err, Error::ContextLengthExceeded(_)), "{err:?}");
        let err = plan_b(Some(8))
            .prompt("<bos>hello world".into(), Some(6))
            .unwrap_err();
        assert!(matches!(err, Error::ContextLengthExceeded(_)), "{err:?}");
        assert!(plan_b(Some(8))
            .prompt("<bos>hello world".into(), Some(5))
            .is_ok());
    }

    #[test]
    fn a_missing_tokenizer_fails_to_load_with_a_reason() {
        let dir = std::env::temp_dir().join(format!("rkmodel-generate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let template = dir.join("chat_template.jinja");
        std::fs::write(&template, "{{ messages[0].content }}").unwrap();

        let mut config = model_config(template.to_str(), None);
        config.tokenizer = Some(dir.join("tokenizer.json"));
        let err = load_err(&config);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(format!("{err:#}").contains("tokenizer.json"), "{err:#}");
    }

    #[test]
    fn a_model_with_no_template_fails_to_load_with_a_reason() {
        let err = load_err(&model_config(None, None));
        assert!(err.to_string().contains("no chat_template"), "{err}");
    }

    #[test]
    fn a_missing_template_file_fails_to_load_with_a_reason() {
        let err = load_err(&model_config(Some("/nope/chat_template.jinja"), None));
        assert!(err.to_string().contains("reading chat template"), "{err}");
    }

    #[test]
    fn a_model_with_no_markers_never_reasons() {
        let m = GenerateModel {
            template: ChatTemplate::from_source("x".into(), Default::default()).unwrap(),
            reasoning: None,
            sampling: Sampling::default(),
            max_context_len: None,
            max_new_tokens: None,
            tokenizer: None,
            tool_format: None,
        };
        assert!(!m.wants_reasoning(None));
        // A client asking a non-reasoning model to reason is a hint, not an
        // error, so it is accepted and ignored.
        assert!(!m.wants_reasoning(Some(true)));
        assert!(!m.parser("", true).in_reasoning());
    }

    #[test]
    fn the_request_overrides_the_configured_default() {
        let mut cfg = markers();
        cfg.default = true;
        let m = GenerateModel {
            template: ChatTemplate::from_source("x".into(), Default::default()).unwrap(),
            reasoning: Some(ReasoningMarkers {
                start: cfg.start.clone(),
                end: cfg.end.clone(),
                default_on: cfg.default,
            }),
            sampling: Sampling::default(),
            max_context_len: None,
            max_new_tokens: None,
            tokenizer: None,
            tool_format: None,
        };
        assert!(m.wants_reasoning(None), "absent takes the model's default");
        assert!(!m.wants_reasoning(Some(false)));
        assert!(m.wants_reasoning(Some(true)));
    }

    #[test]
    fn a_prompt_that_opens_the_block_starts_the_parser_inside_it() {
        let m = GenerateModel {
            template: ChatTemplate::from_source("x".into(), Default::default()).unwrap(),
            reasoning: Some(ReasoningMarkers {
                start: "<think>".into(),
                end: "</think>".into(),
                default_on: true,
            }),
            sampling: Sampling::default(),
            max_context_len: None,
            max_new_tokens: None,
            tokenizer: None,
            tool_format: None,
        };
        assert!(m
            .parser("<|im_start|>assistant\n<think>\n", true)
            .in_reasoning());
        assert!(!m.parser("<|im_start|>assistant\n", true).in_reasoning());
    }
}
