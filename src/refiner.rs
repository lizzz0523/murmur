use std::num::NonZeroU32;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};

const CONTEXT_SIZE: u32 = 2048;
const MAX_NEW_TOKENS: usize = 256;
const SYSTEM_PROMPT: &str = "将用户口述内容整理为通顺、正式的书面文本：去掉口头语、重复与语气词，补全标点。严格保持原意，不添加、不删减、不解释。可以纠正明显的同音字、近音字、专有名词和技术术语的识别错误，只在有把握时纠正，不确定则保留原文。必须使用与输入完全相同的语言输出，不得翻译或更改语言，也不要引入输入之外的其他语言。只输出整理后的文本。";

static LLAMA_BACKEND: LazyLock<LlamaBackend> =
    LazyLock::new(|| LlamaBackend::init().expect("failed to init llama backend"));

pub(crate) struct Refiner {
    model: LlamaModel,
    template: LlamaChatTemplate,
}

impl Refiner {
    pub(crate) fn load(model_path: &Path) -> anyhow::Result<Self> {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

        let model = LlamaModel::load_from_file(
            &LLAMA_BACKEND,
            model_path,
            &LlamaModelParams::default().with_n_gpu_layers(999),
        )
        .context("failed to load refiner model")?;

        let template = model
            .chat_template(None)
            .context("refiner model has no chat template")?;

        Ok(Self { model, template })
    }

    pub(crate) fn refine(&self, content: &str) -> anyhow::Result<String> {
        let messages = [
            LlamaChatMessage::new("system".to_string(), SYSTEM_PROMPT.to_string())?,
            LlamaChatMessage::new("user".to_string(), format!("{content}\n/no_think"))?,
        ];
        let prompt = self
            .model
            .apply_chat_template(&self.template, &messages, true)?;

        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;
        let max_tokens = CONTEXT_SIZE as usize - MAX_NEW_TOKENS;
        if tokens.len() > max_tokens {
            bail!("refiner prompt exceeds context window");
        }

        let mut context = self.model.new_context(
            &LLAMA_BACKEND,
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(CONTEXT_SIZE)),
        )?;

        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        batch.add_sequence(&tokens, 0, false)?;
        context.decode(&mut batch)?;

        let mut sampler = LlamaSampler::greedy();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();

        for pos in (tokens.len() as i32..).take(MAX_NEW_TOKENS) {
            let token = sampler.sample(&context, -1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            output.push_str(
                &self
                    .model
                    .token_to_piece(token, &mut decoder, false, None)?,
            );
            batch.clear();
            batch.add(token, pos, &[0], true)?;
            context.decode(&mut batch)?;
        }

        let refined = strip_reasoning(&output).trim();
        if refined.is_empty() {
            bail!("refiner returned empty output");
        }
        Ok(refined.to_string())
    }
}

fn strip_reasoning(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(idx) => &text[idx + "</think>".len()..],
        None => text,
    }
}
