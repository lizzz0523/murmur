use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, anyhow};
use hf_hub::HFClient;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use mistralrs::{GgufModelBuilder, Model as LLMModel, TextMessageRole, TextMessages};
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
const SYSTEM_PROMPT: &str = "你是语音转写文本的校对工具。输入是语音识别得到的口述文本，可能是中文、英文或中英混合，可能含错别字、重复、语气词或缺少标点。只做清理：删除语气词和重复、纠正明显的错别字、补全标点，逐词保持原语言不变，绝不翻译。数字、字母、型号和专有名词必须保持原有书面形态：阿拉伯数字仍写作阿拉伯数字，英文大小写和连字符原样保留，绝不把数字或英文改写成中文汉字或中文大写。";
const EXAMPLE_INPUT: &str = "這個 feature 的 deadline 是下週五，麻煩先 submit 一個 pull request";
const EXAMPLE_OUTPUT: &str = "这个 feature 的 deadline 是下周五，麻烦先 submit 一个 pull request。";
const EXAMPLE_INPUT_2: &str = "我們用千问三杠一点七 B做校對，二零二四年的資料也要一起測";
const EXAMPLE_OUTPUT_2: &str = "我们用 Qwen3-1.7B 做校对，2024 年的数据也要一起测。";

struct RecognizerInner {
    llm: LLMModel,
    asr: OfflineRecognizer,
    vad: VoiceActivityDetector,
    denoiser: OfflineSpeechDenoiser,
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
                .progress(PrintProgressHandler)
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
                .progress(PrintProgressHandler)
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
                .progress(PrintProgressHandler)
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

        let llm = {
            let repos = client.model("Qwen", "Qwen3-1.7B-GGUF");
            let _ = repos
                .snapshot_download()
                .allow_patterns(vec!["Qwen3-1.7B-Q8_0.gguf".to_string()])
                .max_workers(3)
                .progress(PrintProgressHandler)
                .send()
                .await?;
            GgufModelBuilder::new("Qwen/Qwen3-1.7B-GGUF", vec!["Qwen3-1.7B-Q8_0.gguf"])
                .build()
                .await?
        };

        Ok(Self {
            llm,
            asr,
            vad,
            denoiser,
        })
    }

    async fn run(&self, samples: &[f32], sample_rate: u32) -> String {
        let samples = audio::resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE);
        let samples = audio::high_pass(&samples, TARGET_SAMPLE_RATE, HIGH_PASS_HZ);
        let samples = audio::normalize(&samples, TARGET_RMS_DBFS, MAX_GAIN_DB);

        let mut denoised = self.denoise(&samples);
        audio::limit_peak(&mut denoised, PEAK_CEILING_DBFS);

        let samples = self.filter(&denoised);
        let content = self.recognize(samples);

        self.polish(&content).await.unwrap_or(content)
    }

    async fn polish(&self, content: &str) -> anyhow::Result<String> {
        let messages = TextMessages::new()
            .enable_thinking(false)
            .add_message(TextMessageRole::System, SYSTEM_PROMPT)
            .add_message(TextMessageRole::User, EXAMPLE_INPUT)
            .add_message(TextMessageRole::Assistant, EXAMPLE_OUTPUT)
            .add_message(TextMessageRole::User, EXAMPLE_INPUT_2)
            .add_message(TextMessageRole::Assistant, EXAMPLE_OUTPUT_2)
            .add_message(TextMessageRole::User, content);

        let response = self.llm.send_chat_request(messages).await?;

        Ok(response.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or_default()
            .trim()
            .to_string())
    }

    fn recognize(&self, samples: &[f32]) -> String {
        let stream = self.asr.create_stream();
        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);

        self.asr.decode(&stream);

        stream
            .get_result()
            .map(|result| result.text.trim().to_string())
            .unwrap_or_default()
    }

    fn filter<'a>(&self, samples: &'a [f32]) -> &'a [f32] {
        self.vad.reset();

        let mut bounds: Option<(usize, usize)> = None;
        let mut collect_bounds = || {
            while let Some(segment) = self.vad.front() {
                let start = segment.start().max(0) as usize;
                let end = start + segment.n().max(0) as usize;
                bounds = Some(match &bounds {
                    None => (start, end),
                    Some(prev) => (prev.0.min(start), prev.1.max(end)),
                });
                self.vad.pop();
            }
        };

        for chunk in samples.chunks(512) {
            self.vad.accept_waveform(chunk);
            collect_bounds();
        }
        self.vad.flush();
        collect_bounds();

        match bounds {
            Some((start, end)) => {
                let margin = (0.8 * TARGET_SAMPLE_RATE as f32) as usize;
                let start = start.saturating_sub(margin);
                let end = (end + margin).min(samples.len());
                &samples[start..end]
            }
            None => samples,
        }
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

struct PrintProgressHandler;

impl ProgressHandler for PrintProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start {
                total_files,
                total_bytes,
            } => {
                println!("start download models, total files: {total_files} bytes: {total_bytes}");
            }
            DownloadEvent::Complete => {
                println!("models download complete");
            }
            _ => {}
        }
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
