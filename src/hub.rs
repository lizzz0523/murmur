use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Context;
use hf_hub::HFClient;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use indicatif::{ProgressBar, ProgressStyle};

pub struct Model {
    pub denoiser: PathBuf,
    pub asr_conv_frontend: PathBuf,
    pub asr_encoder: PathBuf,
    pub asr_decoder: PathBuf,
    pub asr_tokenizer: PathBuf,
    pub vad: PathBuf,
    pub refiner: PathBuf,
}

pub async fn resolve_model() -> anyhow::Result<Model> {
    let client = HFClient::new()?;

    let denoiser = client
        .model("csukuangfj", "speech-enhancement-models")
        .snapshot_download()
        .allow_patterns(vec!["gtcrn_simple.onnx".to_string()])
        .max_workers(1)
        .progress(PrintProgressHandler::new("gtcrn"))
        .send()
        .await
        .with_context(|| "failed to download csukuangfj/speech-enhancement-models")?;

    let asr = client
        .model("solavr", "sherpa-onnx-qwen3-asr-1.7B-int8")
        .snapshot_download()
        .allow_patterns(vec![
            "conv_frontend.onnx".to_string(),
            "encoder.int8.onnx".to_string(),
            "decoder.int8.onnx".to_string(),
            "decoder.int8.onnx.data".to_string(),
            "tokenizer/*".to_string(),
        ])
        .max_workers(3)
        .progress(PrintProgressHandler::new("qwen3-asr-1.7B"))
        .send()
        .await
        .with_context(|| "failed to download solavr/sherpa-onnx-qwen3-asr-1.7B-int8")?;

    let vad = client
        .model("csukuangfj", "vad")
        .snapshot_download()
        .allow_patterns(vec!["silero_vad_v5.onnx".to_string()])
        .max_workers(3)
        .progress(PrintProgressHandler::new("silero-vad-v5"))
        .send()
        .await
        .with_context(|| "failed to download csukuangfj/vad")?;

    let refiner = client
        .model("unsloth", "Qwen3-4B-GGUF")
        .snapshot_download()
        .allow_patterns(vec!["Qwen3-4B-Q4_K_M.gguf".to_string()])
        .max_workers(1)
        .progress(PrintProgressHandler::new("qwen3-4b-gguf"))
        .send()
        .await
        .with_context(|| "failed to download unsloth/Qwen3-4B-GGUF")?;

    Ok(Model {
        denoiser: denoiser.join("gtcrn_simple.onnx"),
        asr_conv_frontend: asr.join("conv_frontend.onnx"),
        asr_encoder: asr.join("encoder.int8.onnx"),
        asr_decoder: asr.join("decoder.int8.onnx"),
        asr_tokenizer: asr.join("tokenizer"),
        vad: vad.join("silero_vad_v5.onnx"),
        refiner: refiner.join("Qwen3-4B-Q4_K_M.gguf"),
    })
}

struct PrintProgressHandler {
    bar: ProgressBar,
    files: Mutex<HashMap<String, u64>>,
}

impl PrintProgressHandler {
    fn new(model: &'static str) -> Self {
        const BAR_WIDTH: usize = 24;
        const LABEL_WIDTH: usize = 22;

        let bar = ProgressBar::new(0);
        bar.set_style(
            ProgressStyle::with_template(&format!(
                "{{spinner:.green}} {{prefix:<{LABEL_WIDTH}}} [{{bar:{BAR_WIDTH}.cyan/blue}}] {{bytes}}/{{total_bytes}}"
            ))
            .expect("invalid progress template")
            .progress_chars("=> "),
        );
        bar.set_prefix(model);

        Self {
            bar,
            files: Mutex::new(HashMap::new()),
        }
    }

    fn set_position(&self, position: u64) {
        if position > self.bar.position() {
            self.bar.set_position(position);
        }
    }

    fn set_length(&self, length: u64) {
        if self.bar.length().is_none_or(|current| length > current) {
            self.bar.set_length(length);
        }
    }
}

impl ProgressHandler for PrintProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };

        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.files
                    .lock()
                    .unwrap_or_else(|err| err.into_inner())
                    .clear();
                self.set_length(*total_bytes);
            }
            DownloadEvent::Progress { files } => {
                let sum = {
                    let mut tracked = self.files.lock().unwrap_or_else(|err| err.into_inner());
                    for file in files {
                        let entry = tracked.entry(file.filename.clone()).or_insert(0);
                        *entry = (*entry).max(file.bytes_completed);
                    }
                    tracked.values().copied().sum()
                };
                self.set_position(sum);
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                self.set_length(*total_bytes);
                self.set_position(*bytes_completed);
            }
            DownloadEvent::Complete => self.bar.finish(),
        }
    }
}
