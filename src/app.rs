use std::time::Duration;

use eframe::egui;
use enigo::{Enigo, Keyboard};

use crate::hotkey::Hotkey;
use crate::recognizer::{ReadyHook, Recognizer};
use crate::recorder::Recorder;

enum State {
    Loading(ReadyHook),
    Ready,
    Recording,
    Recognizing,
    Failed,
}

pub struct App {
    hotkey: Hotkey,
    recorder: Recorder,
    recognizer: Recognizer,
    state: State,
    enigo: Enigo,
}

impl App {
    pub fn new(ctx: &egui::Context) -> Self {
        let (recognizer, ready_hook) = Recognizer::load().unwrap();

        Self {
            hotkey: Hotkey::new(ctx).unwrap(),
            recorder: Recorder::new(),
            recognizer,
            state: State::Loading(ready_hook),
            enigo: Enigo::new(&enigo::Settings::default()).unwrap(),
        }
    }

    fn start_record(&mut self) {
        if !matches!(self.state, State::Ready) {
            return;
        }
        self.recorder.start();
        self.state = State::Recording;
    }

    fn stop_record(&mut self) {
        if !matches!(self.state, State::Recording) {
            return;
        }
        let samples = self.recorder.stop();
        let sample_rate = self.recorder.sample_rate();
        if !samples.is_empty() {
            self.recognizer.run(samples, sample_rate);
            self.state = State::Recognizing;
        } else {
            self.state = State::Ready;
        }
    }

    fn paste_text(&mut self, text: &str) {
        if !matches!(self.state, State::Recognizing) {
            return;
        }
        let _ = self.enigo.text(text);
        self.state = State::Ready;
    }

    fn poll_state(&mut self, ctx: &egui::Context) {
        self.recorder.poll();

        match &self.state {
            State::Loading(ready_hook) => match ready_hook.poll() {
                Ok(ready) => {
                    if ready {
                        self.state = State::Ready
                    }
                }
                Err(_) => self.state = State::Failed,
            },
            State::Ready => {
                if self.hotkey.is_pressed() {
                    self.start_record();
                }
            }
            State::Recording => {
                if !self.hotkey.is_pressed() {
                    self.stop_record();
                }
            }
            State::Recognizing => {
                if let Some(text) = self.recognizer.poll() {
                    self.paste_text(&text);
                }
            }
            _ => {}
        }
        if matches!(self.state, State::Recognizing | State::Recording) {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new().show(ui, |ui| {
            let rect = ui.max_rect();
            let (_response, painter) = ui.allocate_painter(rect.size(), egui::Sense::click());

            painter.rect(
                rect,
                rect.height() / 2.0,
                egui::Color32::from_rgb(12, 12, 12),
                (1.0, egui::Color32::from_rgb(45, 45, 45)),
                egui::StrokeKind::Inside,
            );

            match &self.state {
                State::Loading(_) => self.draw_loading(ui),
                State::Failed => self.draw_text(ui, "Error"),
                State::Ready => self.draw_text(ui, "Ready"),
                State::Recording => self.draw_recording(ui),
                State::Recognizing => self.draw_text(ui, "Thinking"),
            }
        });
    }

    fn draw_text(&mut self, ui: &mut egui::Ui, text: &str) {
        let rect = ui.max_rect();

        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
    }

    fn draw_loading(&mut self, ui: &mut egui::Ui) {
        const SPINNER: f32 = 14.0;
        const LEFT_PAD: f32 = 8.0;

        let color = egui::Color32::WHITE;
        let rect = ui.max_rect();
        let center_y = rect.center().y;

        let spinner_rect = egui::Rect::from_center_size(
            egui::pos2(rect.left() + LEFT_PAD + SPINNER / 2.0, center_y),
            egui::vec2(SPINNER, SPINNER),
        );
        ui.put(
            spinner_rect,
            egui::Spinner::new().size(SPINNER).color(color),
        );

        let text_left = spinner_rect.right();
        let text_center_x = (text_left + rect.right()) / 2.0;
        let galley = ui.painter().layout_no_wrap(
            "Loading".to_owned(),
            egui::FontId::proportional(12.0),
            color,
        );
        ui.painter().galley(
            egui::pos2(
                text_center_x - galley.size().x / 2.0,
                center_y - galley.size().y / 2.0,
            ),
            galley,
            color,
        );
    }

    fn draw_recording(&mut self, ui: &mut egui::Ui) {
        const N: usize = 10;
        const BAR_W: f32 = 2.5;
        const GAP: f32 = 3.5;
        const MIN_H: f32 = 2.5;
        const MAX_H: f32 = 19.0;
        const WEIGHTS: [f32; N] = [0.16, 0.25, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.25, 0.16];

        let total_w = N as f32 * BAR_W + (N as f32 - 1.0) * GAP;
        let rect = ui.max_rect();
        let start_x = rect.center().x - total_w / 2.0;
        let center_y = rect.center().y;

        let dbfs = self.recorder.dbfs();
        let level = ((60.0 + dbfs) / 60.0).clamp(0.0, 1.0);

        for (i, weight) in WEIGHTS.iter().enumerate() {
            let h = (MIN_H + (MAX_H - MIN_H) * level * weight).max(BAR_W);
            let x = start_x + i as f32 * (BAR_W + GAP) + BAR_W / 2.0;
            let bar = egui::Rect::from_center_size(egui::pos2(x, center_y), egui::vec2(BAR_W, h));
            ui.painter().rect(
                bar,
                BAR_W / 2.0,
                egui::Color32::WHITE,
                egui::Stroke::NONE,
                egui::StrokeKind::Inside,
            );
        }
    }
}

impl eframe::App for App {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        egui::Rgba::TRANSPARENT.to_array()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_state(ui.ctx());
        self.draw_ui(ui);
    }
}

pub fn get_window_size() -> (f32, f32) {
    (100.0, 33.0)
}
