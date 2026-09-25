use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail};
use eframe::egui;
use sherpa_onnx::Wave;

mod app;
use app::{App, get_window_size};

mod recognizer;
use recognizer::Recognizer;

mod pipeline;
mod recorder;
mod refiner;

mod audio;
mod hotkey;
mod tray;

mod hub;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(path) = &args.file {
        run_file(path)?;
        return Ok(());
    }
    run()
}

struct Args {
    file: Option<String>,
}

impl Args {
    fn parse() -> Self {
        let mut file = None;
        for arg in std::env::args().skip(1) {
            if arg.starts_with('-') {
                continue;
            }
            if file.is_none() {
                file = Some(arg);
            }
        }
        Self { file }
    }
}

fn run_file(path: &str) -> anyhow::Result<()> {
    let wave = Wave::read(path).ok_or_else(|| anyhow!("failed to read WAV: {path}"))?;
    if wave.num_samples() == 0 {
        bail!("empty audio: {path}");
    }

    let (recognizer, ready) = Recognizer::load()?;
    loop {
        match ready.poll()? {
            true => break,
            false => thread::sleep(Duration::from_millis(50)),
        }
    }

    recognizer.begin();
    recognizer.push(wave.samples().to_vec(), wave.sample_rate() as u32);
    recognizer.end();

    if let Some(text) = recognizer.recv() {
        println!("{text}");
    }

    Ok(())
}

fn run() -> anyhow::Result<()> {
    let (sw, sh) = get_resolution();
    let (w, h) = get_window_size();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_transparent(true)
            .with_decorations(false)
            .with_movable_by_background(true)
            .with_always_on_top()
            .with_inner_size([w, h])
            .with_position(((sw - w) / 2.0, sh - h - 40.0))
            .with_icon(load_icon())
            .with_resizable(false),
        ..Default::default()
    };
    eframe::run_native(
        "murmur",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(&cc.egui_ctx)))),
    )?;

    Ok(())
}

fn load_icon() -> egui::IconData {
    eframe::icon_data::from_png_bytes(include_bytes!("../assets/appicon.png"))
        .expect("failed to load app icon")
}

fn get_resolution() -> (f32, f32) {
    use core_graphics::display::{CGDisplayBounds, CGMainDisplayID};

    let bounds = unsafe { CGDisplayBounds(CGMainDisplayID()) };
    let width = bounds.size.width;
    let height = bounds.size.height;

    (width as f32, height as f32)
}
