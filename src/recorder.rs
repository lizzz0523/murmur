use std::sync::mpsc;

use anyhow::{anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::audio;

pub struct InputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

pub struct Recorder {
    host: cpal::Host,
    tx: mpsc::Sender<Vec<f32>>,
    rx: mpsc::Receiver<Vec<f32>>,
    samples: Vec<f32>,
    sample_rate: u32,
    smooth_rms: f32,
    stream: cpal::Stream,
    device_id: String,
}

impl Recorder {
    pub fn new() -> anyhow::Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .or_else(|| host.input_devices().ok()?.find(|d| d.supports_input()))
            .ok_or_else(|| anyhow!("no input device available"))?;

        let (tx, rx) = mpsc::channel();
        let (stream, sample_rate) = build_stream(&device, &tx)?;
        let device_id = device.id().map(|id| id.to_string()).unwrap_or_default();

        Ok(Self {
            host,
            tx,
            rx,
            samples: Vec::new(),
            sample_rate,
            smooth_rms: 0.0,
            stream,
            device_id,
        })
    }

    pub fn current_device(&self) -> &str {
        &self.device_id
    }

    pub fn list_devices(&self) -> Vec<InputDevice> {
        let default_id = self.host.default_input_device().and_then(|d| d.id().ok());

        self.host
            .input_devices()
            .map(|devices| {
                devices
                    .filter(|device| device.supports_input())
                    .map(|device| {
                        let id = device.id().ok();
                        InputDevice {
                            name: device.to_string(),
                            is_default: id.as_ref() == default_id.as_ref(),
                            id: id.map(|id| id.to_string()).unwrap_or_default(),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn select_device(&mut self, id: &str) -> anyhow::Result<()> {
        let device_id: cpal::DeviceId = id.parse()?;
        let device = self
            .host
            .device_by_id(&device_id)
            .ok_or_else(|| anyhow!("input device not found: {id}"))?;

        let _ = self.stream.pause();
        let (stream, sample_rate) = build_stream(&device, &self.tx)?;

        self.stream = stream;
        self.sample_rate = sample_rate;
        self.device_id = id.to_string();
        self.samples.clear();
        self.smooth_rms = 0.0;
        while self.rx.try_recv().is_ok() {}

        Ok(())
    }

    pub fn start(&mut self) -> anyhow::Result<()> {
        while self.rx.try_recv().is_ok() {}
        self.stream.play()?;
        Ok(())
    }

    pub fn stop(&mut self) -> anyhow::Result<Vec<f32>> {
        self.stream.pause()?;
        while let Ok(samples) = self.rx.try_recv() {
            self.samples.extend_from_slice(&samples[..]);
        }
        Ok(self.samples.drain(..).collect())
    }

    pub fn poll(&mut self) {
        let mut last_samples = None;
        while let Ok(samples) = self.rx.try_recv() {
            self.samples.extend_from_slice(&samples[..]);
            last_samples = Some(samples);
        }

        let rms = if let Some(samples) = last_samples {
            let sum_sq = samples.iter().map(|s| s * s).sum::<f32>();
            (sum_sq / samples.len() as f32).sqrt()
        } else {
            0.0
        };

        let alpha = 0.9;
        self.smooth_rms = alpha * rms + (1.0 - alpha) * self.smooth_rms;
    }

    pub fn dbfs(&mut self) -> f32 {
        if self.smooth_rms < 1e-9 {
            -100.0
        } else {
            20.0 * self.smooth_rms.log10()
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

fn build_stream(
    device: &cpal::Device,
    tx: &mpsc::Sender<Vec<f32>>,
) -> anyhow::Result<(cpal::Stream, u32)> {
    let config = device.default_input_config()?;
    let channels = config.channels();
    let sample_rate = config.sample_rate();

    if config.sample_format() != cpal::SampleFormat::F32 {
        bail!("unsupported sample format: {:?}", config.sample_format());
    }

    let tx = tx.clone();
    let stream = device.build_input_stream(
        config.config(),
        move |data: &[f32], _info| {
            let _ = tx.send(audio::downmix(data, channels));
        },
        |err| eprintln!("input stream error: {err}"),
        None,
    )?;

    Ok((stream, sample_rate))
}
