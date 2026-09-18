use std::f32::consts::{FRAC_1_SQRT_2, PI};

pub fn resample_linear(samples: &[f32], sample_rate: u32, output_sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() || sample_rate == output_sample_rate {
        return samples.to_vec();
    }
    let ratio = output_sample_rate as f64 / sample_rate as f64;
    let output_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src = i as f64 / ratio;
        let index = src.floor() as usize;
        let frac = (src - index as f64) as f32;
        let a = samples.get(index).copied().unwrap_or(0.0);
        let b = samples.get(index + 1).copied().unwrap_or(a);
        output.push(a + (b - a) * frac);
    }
    output
}

pub fn high_pass(samples: &[f32], sample_rate: u32, cutoff_hz: f32) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
    }

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

    let b0 = b0 / a0;
    let b1 = b1 / a0;
    let b2 = b2 / a0;
    let a1 = a1 / a0;
    let a2 = a2 / a0;

    let mut x1 = 0.0;
    let mut x2 = 0.0;
    let mut y1 = 0.0;
    let mut y2 = 0.0;

    let mut output = Vec::with_capacity(samples.len());
    for &x in samples {
        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        output.push(y);
    }
    output
}

pub fn normalize(samples: &[f32], target_rms_dbfs: f32, max_gain_db: f32) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
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
        return samples.to_vec();
    }

    frame_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((frame_rms.len() as f32 - 1.0) * 0.95).round() as usize;
    let speech_rms = frame_rms[idx];

    if speech_rms < 1e-6 {
        return samples.to_vec();
    }

    let target = 10f32.powf(target_rms_dbfs / 20.0);
    let max_gain = 10f32.powf(max_gain_db / 20.0);
    let gain = (target / speech_rms).min(max_gain);

    samples.iter().map(|s| s * gain).collect()
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
