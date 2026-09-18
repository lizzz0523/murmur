use std::collections::HashMap;
use std::iter;
use std::ops::Range;
use std::path::Path;
use std::sync::{Mutex, mpsc};

use anyhow::{Context, anyhow};
use hf_hub::HFClient;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use indicatif::{ProgressBar, ProgressStyle};
use mistralrs::{Model as LLMModel, TextMessageRole, TextMessages, TextModelBuilder};
use sherpa_onnx::{
    OfflineModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSpeechDenoiser, OfflineSpeechDenoiserConfig, OfflineSpeechDenoiserGtcrnModelConfig,
    OfflineSpeechDenoiserModelConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use tokio::sync::mpsc as tokio_mpsc;

use crate::audio;

pub struct ReadyHook(mpsc::Receiver<anyhow::Result<()>>);

impl ReadyHook {
    pub fn poll(&self) -> anyhow::Result<bool> {
        match self.0.try_recv() {
            Ok(Ok(())) => Ok(true),
            Ok(Err(err)) => Err(err),
            Err(mpsc::TryRecvError::Disconnected) => Err(anyhow!("ready hook internal error")),
            Err(mpsc::TryRecvError::Empty) => Ok(false),
        }
    }
}

enum RecognizerCall {
    Run { samples: Vec<f32>, sample_rate: u32 },
}

pub struct Recognizer {
    tx: tokio_mpsc::UnboundedSender<RecognizerCall>,
    rx: mpsc::Receiver<String>,
    _rt: tokio::runtime::Runtime,
}

impl Recognizer {
    pub fn load() -> anyhow::Result<(Self, ReadyHook)> {
        let rt = tokio::runtime::Runtime::new()?;
        let (tx, mut rx_spawn) = tokio_mpsc::unbounded_channel();
        let (tx_spawn, rx) = mpsc::channel();
        let (tx_ready, rx_ready) = mpsc::channel();

        rt.spawn(async move {
            let inner = match RecognizerInner::async_load().await {
                Ok(inner) => inner,
                Err(err) => {
                    eprintln!("recognizer load failed: {err:#}");
                    let _ = tx_ready.send(Err(err));
                    return;
                }
            };
            let _ = tx_ready.send(Ok(()));

            while let Some(call) = rx_spawn.recv().await {
                match call {
                    RecognizerCall::Run {
                        samples,
                        sample_rate,
                    } => {
                        let result = inner.run(&samples, sample_rate).await;
                        let _ = tx_spawn.send(result);
                    }
                }
            }
        });

        Ok((Self { _rt: rt, tx, rx }, ReadyHook(rx_ready)))
    }

    pub fn run(&self, samples: Vec<f32>, sample_rate: u32) {
        let _ = self.tx.send(RecognizerCall::Run {
            samples,
            sample_rate,
        });
    }

    pub fn poll(&self) -> Option<String> {
        self.rx.try_recv().ok()
    }
}

const TARGET_SAMPLE_RATE: u32 = 16_000;
const HIGH_PASS_HZ: f32 = 100.0;
const TARGET_RMS_DBFS: f32 = -20.0;
const MAX_GAIN_DB: f32 = 26.0;
const PEAK_CEILING_DBFS: f32 = -1.0;
const SEGMENT_MARGIN_SECONDS: f32 = 0.8;
const MERGE_GAP_SECONDS: f32 = 0.6;
const MAX_CHUNK_SECONDS: f32 = 20.0;
const SYSTEM_PROMPT: &str =
    "将中文口语转写改写为正式、自然的书面语。保持原意，不添加原文没有的信息，只输出改写后的文本。";

struct RecognizerInner {
    denoiser: OfflineSpeechDenoiser,
    asr: OfflineRecognizer,
    vad: VoiceActivityDetector,
    refiner: LLMModel,
}

impl RecognizerInner {
    async fn async_load() -> anyhow::Result<Self> {
        let client = HFClient::new()?;

        let denoiser = {
            let repos = client.model("csukuangfj", "speech-enhancement-models");
            let downloaded = repos
                .snapshot_download()
                .allow_patterns(vec!["gtcrn_simple.onnx".to_string()])
                .max_workers(1)
                .progress(PrintProgressHandler::new("gtcrn"))
                .send()
                .await?;
            let config = OfflineSpeechDenoiserConfig {
                model: OfflineSpeechDenoiserModelConfig {
                    gtcrn: OfflineSpeechDenoiserGtcrnModelConfig {
                        model: Some(path_string(&downloaded.join("gtcrn_simple.onnx"))),
                    },
                    num_threads: 1,
                    provider: Some("cpu".to_string()),
                    ..Default::default()
                },
            };
            let denoiser = OfflineSpeechDenoiser::create(&config);
            denoiser.with_context(|| "failed to create speech denoiser")?
        };

        let asr = {
            let repos = client.model("csukuangfj2", "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25");
            let downloaded = repos
                .snapshot_download()
                .allow_patterns(vec![
                    "conv_frontend.onnx".to_string(),
                    "encoder.int8.onnx".to_string(),
                    "decoder.int8.onnx".to_string(),
                    "tokenizer/*".to_string(),
                ])
                .max_workers(3)
                .progress(PrintProgressHandler::new("qwen3-asr-0.6B"))
                .send()
                .await?;
            let config = OfflineRecognizerConfig {
                model_config: OfflineModelConfig {
                    qwen3_asr: OfflineQwen3ASRModelConfig {
                        conv_frontend: Some(path_string(&downloaded.join("conv_frontend.onnx"))),
                        encoder: Some(path_string(&downloaded.join("encoder.int8.onnx"))),
                        decoder: Some(path_string(&downloaded.join("decoder.int8.onnx"))),
                        tokenizer: Some(path_string(&downloaded.join("tokenizer"))),
                        max_new_tokens: 512,
                        ..Default::default()
                    },
                    num_threads: 8,
                    ..Default::default()
                },
                ..Default::default()
            };
            let asr = OfflineRecognizer::create(&config);
            asr.with_context(|| "failed to create recognizer")?
        };

        let vad = {
            let repos = client.model("csukuangfj", "vad");
            let downloaded = repos
                .snapshot_download()
                .allow_patterns(vec!["silero_vad_v5.onnx".to_string()])
                .max_workers(3)
                .progress(PrintProgressHandler::new("silero-vad-v5"))
                .send()
                .await?;
            let config = VadModelConfig {
                silero_vad: SileroVadModelConfig {
                    model: Some(path_string(&downloaded.join("silero_vad_v5.onnx"))),
                    threshold: 0.5,
                    min_silence_duration: 0.25,
                    min_speech_duration: 0.25,
                    max_speech_duration: 5.0,
                    ..Default::default()
                },
                sample_rate: TARGET_SAMPLE_RATE as i32,
                num_threads: 1,
                provider: Some("cpu".to_string()),
                ..Default::default()
            };
            let vad = VoiceActivityDetector::create(&config, 30.0);
            vad.with_context(|| "failed to create recognizer")?
        };

        let refiner = {
            let repos = client.model("Aye10032", "Qwen3-ASR-Refiner-0.6B");
            let downloaded = repos
                .snapshot_download()
                .allow_patterns(vec![
                    "config.json".to_string(),
                    "generation_config.json".to_string(),
                    "chat_template.jinja".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "model.safetensors".to_string(),
                ])
                .max_workers(3)
                .progress(PrintProgressHandler::new("qwen3-asr-refiner-0.6b"))
                .send()
                .await?;
            TextModelBuilder::new(path_string(&downloaded))
                .build()
                .await?
        };

        Ok(Self {
            denoiser,
            asr,
            vad,
            refiner,
        })
    }

    async fn run(&self, samples: &[f32], sample_rate: u32) -> String {
        let samples = audio::resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE);
        let samples = audio::high_pass(&samples, TARGET_SAMPLE_RATE, HIGH_PASS_HZ);
        let samples = audio::normalize(&samples, TARGET_RMS_DBFS, MAX_GAIN_DB);

        let mut samples = self.denoise(&samples);
        audio::limit_peak(&mut samples, PEAK_CEILING_DBFS);

        let segments = self.segments(&samples);
        let content = self.recognize(&samples, &segments);

        self.refine(&content).await.unwrap_or(content)
    }

    async fn refine(&self, content: &str) -> anyhow::Result<String> {
        let messages = TextMessages::new()
            .enable_thinking(false)
            .add_message(TextMessageRole::System, SYSTEM_PROMPT)
            .add_message(TextMessageRole::User, content);

        let response = self.refiner.send_chat_request(messages).await?;

        Ok(response.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or_default()
            .trim()
            .to_string())
    }

    fn recognize(&self, samples: &[f32], segments: &[Range<usize>]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for segment in segments {
            let stream = self.asr.create_stream();
            stream.accept_waveform(TARGET_SAMPLE_RATE as i32, &samples[segment.clone()]);

            self.asr.decode(&stream);

            let text = stream
                .get_result()
                .map(|result| result.text.trim().to_string())
                .unwrap_or_default();
            if !text.is_empty() {
                parts.push(text);
            }
        }
        parts.join("")
    }

    fn segments(&self, samples: &[f32]) -> Vec<Range<usize>> {
        self.vad.reset();

        let mut detected: Vec<(usize, usize)> = Vec::new();
        let mut collect_segments = || {
            while let Some(segment) = self.vad.front() {
                let start = segment.start().max(0) as usize;
                let end = start + segment.n().max(0) as usize;
                detected.push((start, end));
                self.vad.pop();
            }
        };

        for chunk in samples.chunks(512) {
            self.vad.accept_waveform(chunk);
            collect_segments();
        }
        self.vad.flush();
        collect_segments();

        if detected.is_empty() {
            return iter::once(0..samples.len()).collect();
        }

        let margin = (SEGMENT_MARGIN_SECONDS * TARGET_SAMPLE_RATE as f32) as usize;
        let merge_gap = (MERGE_GAP_SECONDS * TARGET_SAMPLE_RATE as f32) as usize;
        let max_chunk = (MAX_CHUNK_SECONDS * TARGET_SAMPLE_RATE as f32) as usize;

        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(detected.len());
        for &(start, end) in &detected {
            let start = start.saturating_sub(margin);
            let end = (end + margin).min(samples.len());
            match merged.last_mut() {
                Some(prev) if start <= prev.1 + merge_gap => prev.1 = prev.1.max(end),
                _ => merged.push((start, end)),
            }
        }

        let mut segments = Vec::new();
        for (start, end) in merged {
            let mut chunk_start = start;
            while end - chunk_start > max_chunk {
                segments.push(chunk_start..chunk_start + max_chunk);
                chunk_start += max_chunk;
            }
            if chunk_start < end {
                segments.push(chunk_start..end);
            }
        }

        segments
    }

    fn denoise(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return vec![];
        }
        let result = self.denoiser.run(samples, TARGET_SAMPLE_RATE as i32);
        if result.samples.is_empty() {
            samples.to_vec()
        } else {
            result.samples
        }
    }
}

const BAR_WIDTH: usize = 24;
const LABEL_WIDTH: usize = 22;

struct PrintProgressHandler {
    bar: ProgressBar,
    files: Mutex<HashMap<String, u64>>,
}

impl PrintProgressHandler {
    fn new(model: &'static str) -> Self {
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

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
