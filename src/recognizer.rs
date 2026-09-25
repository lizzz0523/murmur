use std::ops::Range;
use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, anyhow};
use sherpa_onnx::{
    OfflineModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSpeechDenoiser, OfflineSpeechDenoiserConfig, OfflineSpeechDenoiserGtcrnModelConfig,
    OfflineSpeechDenoiserModelConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use tokio::sync::mpsc as tokio_mpsc;

use crate::audio;
use crate::hub;
use crate::refiner::Refiner;

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
                        let result = inner.run(&samples, sample_rate);
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

struct RecognizerInner {
    denoiser: OfflineSpeechDenoiser,
    asr: OfflineRecognizer,
    vad: VoiceActivityDetector,
    refiner: Refiner,
}

impl RecognizerInner {
    async fn async_load() -> anyhow::Result<Self> {
        let model = hub::resolve_model().await?;

        let denoiser = {
            let config = OfflineSpeechDenoiserConfig {
                model: OfflineSpeechDenoiserModelConfig {
                    gtcrn: OfflineSpeechDenoiserGtcrnModelConfig {
                        model: Some(path_string(&model.denoiser)),
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
            let config = OfflineRecognizerConfig {
                model_config: OfflineModelConfig {
                    qwen3_asr: OfflineQwen3ASRModelConfig {
                        conv_frontend: Some(path_string(&model.asr_conv_frontend)),
                        encoder: Some(path_string(&model.asr_encoder)),
                        decoder: Some(path_string(&model.asr_decoder)),
                        tokenizer: Some(path_string(&model.asr_tokenizer)),
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
            let config = VadModelConfig {
                silero_vad: SileroVadModelConfig {
                    model: Some(path_string(&model.vad)),
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

        let refiner = Refiner::create(&model.refiner)?;

        Ok(Self {
            denoiser,
            asr,
            vad,
            refiner,
        })
    }

    fn run(&self, samples: &[f32], sample_rate: u32) -> String {
        let mut samples = audio::resample(samples, sample_rate, TARGET_SAMPLE_RATE);
        audio::high_pass(&mut samples, TARGET_SAMPLE_RATE, HIGH_PASS_HZ);

        let mut samples = self.denoise(&samples);
        let segments = self.segments(&samples);
        if segments.is_empty() {
            return String::new();
        }

        audio::normalize(&mut samples, TARGET_RMS_DBFS, MAX_GAIN_DB);
        audio::limit_peak(&mut samples, PEAK_CEILING_DBFS);

        let content = self.recognize(&samples, &segments);
        if content.trim().is_empty() {
            return String::new();
        }

        self.refiner.refine(&content).unwrap_or(content)
    }

    fn recognize(&self, samples: &[f32], segments: &[Range<usize>]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for segment in segments {
            let stream = self.asr.create_stream();
            stream.accept_waveform(TARGET_SAMPLE_RATE as i32, &samples[segment.clone()]);

            self.asr.decode(&stream);

            let text = stream
                .get_result()
                .map(|result| strip_control_prefix(result.text.trim()))
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
            return Vec::new();
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
            return Vec::new();
        }
        let result = self.denoiser.run(samples, TARGET_SAMPLE_RATE as i32);
        if result.samples.is_empty() {
            samples.to_vec()
        } else {
            result.samples
        }
    }
}

fn strip_control_prefix(text: &str) -> String {
    const MARK: &str = "<asr_text>";
    match text.rfind(MARK) {
        Some(idx) => text[idx + MARK.len()..].trim().to_string(),
        None => text.to_string(),
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
