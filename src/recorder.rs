use std::sync::mpsc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub struct Recorder {
    rx: mpsc::Receiver<Vec<f32>>,
    samples: Vec<f32>,
    sample_rate: u32,
    smooth_rms: f32,
    stream: cpal::Stream,
}

impl Recorder {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();

        let host = cpal::default_host();
        let device = host.default_input_device().unwrap();
        let config = device.default_input_config().unwrap();

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device
                .build_input_stream(
                    config.into(),
                    move |data: &[f32], _info| {
                        let _ = tx.send(data.to_vec());
                    },
                    |_err| {},
                    None,
                )
                .unwrap(),
            _ => unimplemented!(),
        };

        Self {
            rx,
            samples: vec![],
            sample_rate: config.sample_rate(),
            smooth_rms: 0.0,
            stream,
        }
    }

    pub fn start(&mut self) {
        while self.rx.try_recv().is_ok() {}
        self.stream.play().unwrap();
    }

    pub fn stop(&mut self) -> Vec<f32> {
        self.stream.pause().unwrap();
        while let Ok(samples) = self.rx.try_recv() {
            self.samples.extend_from_slice(&samples[..]);
        }
        self.samples.drain(..).collect()
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
