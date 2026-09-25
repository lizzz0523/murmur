use std::mem;
use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, anyhow};
use samplerate::{ConverterType, Samplerate};
use sherpa_onnx::{
    OfflineModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSpeechDenoiser, OfflineSpeechDenoiserConfig, OfflineSpeechDenoiserGtcrnModelConfig,
    OfflineSpeechDenoiserModelConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};

use crate::audio::{self, HighPass};
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
    Begin,
    Push { samples: Vec<f32>, sample_rate: u32 },
    End,
}

pub struct Recognizer {
    tx: mpsc::Sender<RecognizerCall>,
    rx: mpsc::Receiver<String>,
    _rt: tokio::runtime::Runtime,
}

impl Recognizer {
    pub fn load() -> anyhow::Result<(Self, ReadyHook)> {
        let rt = tokio::runtime::Runtime::new()?;
        let (tx, rx_spawn) = mpsc::channel();
        let (tx_spawn, rx) = mpsc::channel();
        let (tx_ready, rx_ready) = mpsc::channel();

        rt.spawn(async move {
            let model = match hub::resolve_model().await {
                Ok(model) => model,
                Err(err) => {
                    eprintln!("model load failed: {err:#}");
                    let _ = tx_ready.send(Err(err));
                    return;
                }
            };

            let result = tokio::task::spawn_blocking(move || {
                let mut inner = match RecognizerInner::new(model) {
                    Ok(inner) => inner,
                    Err(err) => {
                        eprintln!("recognizer load failed: {err:#}");
                        let _ = tx_ready.send(Err(err));
                        return;
                    }
                };
                let _ = tx_ready.send(Ok(()));

                while let Ok(call) = rx_spawn.recv() {
                    match call {
                        RecognizerCall::Begin => {
                            inner.begin();
                        }
                        RecognizerCall::Push {
                            samples,
                            sample_rate,
                        } => {
                            inner.push(&samples, sample_rate);
                        }
                        RecognizerCall::End => {
                            let text = inner.end();
                            let _ = tx_spawn.send(text);
                        }
                    }
                }
            })
            .await;

            if let Err(err) = result {
                eprintln!("recognizer task failed: {err}");
            }
        });

        Ok((Self { _rt: rt, tx, rx }, ReadyHook(rx_ready)))
    }

    pub fn begin(&self) {
        let _ = self.tx.send(RecognizerCall::Begin);
    }

    pub fn push(&self, samples: Vec<f32>, sample_rate: u32) {
        let _ = self.tx.send(RecognizerCall::Push {
            samples,
            sample_rate,
        });
    }

    pub fn end(&self) {
        let _ = self.tx.send(RecognizerCall::End);
    }

    pub fn poll(&self) -> Option<String> {
        self.rx.try_recv().ok()
    }
}

const TARGET_SAMPLE_RATE: u32 = 16_000;
const VAD_CHUNK_SIZE: usize = 512;

const HIGH_PASS_HZ: f32 = 100.0;
const TARGET_RMS_DBFS: f32 = -20.0;
const MAX_GAIN_DB: f32 = 26.0;
const PEAK_CEILING_DBFS: f32 = -1.0;

const SEGMENT_TAIL_MARGIN_SECONDS: f32 = 0.15;

struct RecognizerInner {
    denoiser: OfflineSpeechDenoiser,
    asr: OfflineRecognizer,
    vad: VoiceActivityDetector,
    vad_cursor: usize,
    refiner: Refiner,
    resampler: Option<Samplerate>,
    high_pass: HighPass,
    transcript: String,
    samples: Vec<f32>,
    sample_rate: u32,
    process_end: usize,
}

impl RecognizerInner {
    fn new(model: hub::Model) -> anyhow::Result<Self> {
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
                    max_speech_duration: 10.0,
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
            vad_cursor: 0,
            refiner,
            resampler: None,
            high_pass: HighPass::new(TARGET_SAMPLE_RATE, HIGH_PASS_HZ),
            transcript: String::new(),
            samples: Vec::new(),
            sample_rate: 0,
            process_end: 0,
        })
    }

    fn begin(&mut self) {
        self.vad.reset();
        self.resampler = None;
        self.high_pass = HighPass::new(TARGET_SAMPLE_RATE, HIGH_PASS_HZ);
        self.transcript.clear();
        self.samples.clear();
        self.sample_rate = 0;
        self.vad_cursor = 0;
        self.process_end = 0;
    }

    fn push(&mut self, samples: &[f32], sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        if self.sample_rate != sample_rate {
            self.sample_rate = sample_rate;
            self.resampler = match Samplerate::new(
                ConverterType::SincBestQuality,
                sample_rate,
                TARGET_SAMPLE_RATE,
                1,
            ) {
                Ok(resampler) => Some(resampler),
                Err(err) => {
                    eprintln!("failed to create resampler: {err}");
                    None
                }
            };
        }

        let Some(resampler) = &self.resampler else {
            return;
        };
        let mut samples = match resampler.process(samples) {
            Ok(samples) => samples,
            Err(err) => {
                eprintln!("resample failed: {err}");
                return;
            }
        };
        if samples.is_empty() {
            return;
        }
        self.high_pass.process(&mut samples);
        self.samples.extend_from_slice(&samples);

        while self.samples.len() - self.vad_cursor >= VAD_CHUNK_SIZE {
            let delta = &self.samples[self.vad_cursor..self.vad_cursor + VAD_CHUNK_SIZE];
            self.vad.accept_waveform(delta);
            self.vad_cursor += VAD_CHUNK_SIZE;
        }

        self.collect();
    }

    fn end(&mut self) -> String {
        if let Some(resampler) = &self.resampler {
            match resampler.process_last(&[]) {
                Ok(mut samples) => {
                    self.high_pass.process(&mut samples);
                    self.samples.extend_from_slice(&samples);
                }
                Err(err) => {
                    eprintln!("resample flush failed: {err}");
                }
            }
        }

        if self.vad_cursor < self.samples.len() {
            let delta = &self.samples[self.vad_cursor..];
            self.vad.accept_waveform(delta);
            self.vad_cursor = self.samples.len();
        }
        self.vad.flush();

        self.collect();

        let content = mem::take(&mut self.transcript);
        let content = content.trim();
        if content.is_empty() {
            return String::new();
        }

        self.refiner
            .refine(content)
            .unwrap_or_else(|_| content.to_string())
    }

    fn collect(&mut self) {
        while let Some(segment) = self.vad.front() {
            let start = segment.start().max(0) as usize;
            let end = start + segment.n().max(0) as usize;
            self.vad.pop();
            self.process(start, end);
        }
    }

    fn process(&mut self, start: usize, end: usize) {
        let tail = (SEGMENT_TAIL_MARGIN_SECONDS * TARGET_SAMPLE_RATE as f32) as usize;

        let start = start.max(self.process_end);
        let end = (end + tail).min(self.samples.len());
        if end <= start {
            return;
        }
        self.process_end = end;

        let mut samples = self.denoise(&self.samples[start..end]);
        audio::normalize(&mut samples, TARGET_RMS_DBFS, MAX_GAIN_DB);
        audio::limit_peak(&mut samples, PEAK_CEILING_DBFS);

        let text = self.recognize(&samples);
        if !text.is_empty() {
            self.transcript.push_str(&text);
        }
    }

    fn recognize(&self, samples: &[f32]) -> String {
        let stream = self.asr.create_stream();
        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);

        self.asr.decode(&stream);

        stream
            .get_result()
            .map(|result| strip_control_prefix(result.text.trim()))
            .unwrap_or_default()
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
