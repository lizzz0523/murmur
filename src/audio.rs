use std::f32::consts::{FRAC_1_SQRT_2, PI};

pub struct HighPass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl HighPass {
    pub fn new(sample_rate: u32, cutoff_hz: f32) -> Self {
        let q = FRAC_1_SQRT_2;
        let w0 = 2.0 * PI * cutoff_hz / sample_rate as f32;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let x = *s;
            let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x;
            self.y2 = self.y1;
            self.y1 = y;
            *s = y;
        }
    }
}

pub fn normalize(samples: &mut [f32], target_rms_dbfs: f32, max_gain_db: f32) {
    if samples.is_empty() {
        return;
    }

    let frame_len = (0.02 * 16_000.0) as usize;
    let hop = (frame_len / 2).max(1);

    let mut frame_rms: Vec<f32> = Vec::new();
    let mut start = 0;
    while start + frame_len <= samples.len() {
        let sum_sq = samples[start..start + frame_len]
            .iter()
            .map(|s| s * s)
            .sum::<f32>();
        frame_rms.push((sum_sq / frame_len as f32).sqrt());
        start += hop;
    }

    if frame_rms.is_empty() {
        return;
    }

    frame_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((frame_rms.len() as f32 - 1.0) * 0.95).round() as usize;
    let speech_rms = frame_rms[idx];

    if speech_rms < 1e-6 {
        return;
    }

    let target = 10f32.powf(target_rms_dbfs / 20.0);
    let max_gain = 10f32.powf(max_gain_db / 20.0);
    let gain = (target / speech_rms).min(max_gain);

    for s in samples.iter_mut() {
        *s *= gain;
    }
}

pub fn limit_peak(samples: &mut [f32], ceiling_dbfs: f32) {
    let peak = samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
    if peak <= 1e-9 {
        return;
    }
    let ceiling = 10f32.powf(ceiling_dbfs / 20.0);
    if peak > ceiling {
        let scale = ceiling / peak;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }
}

pub fn downmix(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels < 2 {
        samples.to_vec()
    } else if channels == 2 {
        samples
            .chunks(2)
            .map(|s| s.iter().sum::<f32>() / s.len() as f32)
            .collect()
    } else {
        samples.chunks(channels as usize).map(|s| s[0]).collect()
    }
}
