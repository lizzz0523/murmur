use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use eframe::egui;

pub struct Hotkey {
    pressed: Arc<AtomicBool>,
    _handle: JoinHandle<()>,
}

impl Hotkey {
    pub fn new(ctx: &egui::Context) -> anyhow::Result<Self> {
        let listener = handy_keys::KeyboardListener::new()?;
        let pressed = Arc::new(AtomicBool::new(false));

        let spawn_pressed = pressed.clone();
        let spawn_ctx = ctx.clone();
        let handle = thread::spawn(move || {
            while let Ok(evt) = listener.recv() {
                if evt.is_key_down {
                    if evt.modifiers.contains(handy_keys::Modifiers::CTRL_RIGHT)
                        && !spawn_pressed.load(Ordering::Relaxed)
                    {
                        spawn_pressed.store(true, Ordering::Relaxed);
                        spawn_ctx.request_repaint();
                    }
                } else {
                    if spawn_pressed.load(Ordering::Relaxed)
                        && !evt.modifiers.contains(handy_keys::Modifiers::CTRL_RIGHT)
                    {
                        spawn_pressed.store(false, Ordering::Relaxed);
                        spawn_ctx.request_repaint();
                    }
                }
            }
        });

        Ok(Self {
            pressed,
            _handle: handle,
        })
    }

    pub fn is_pressed(&self) -> bool {
        self.pressed.load(Ordering::Relaxed)
    }
}
