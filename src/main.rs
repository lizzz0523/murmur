use eframe::egui;

mod app;
use app::{App, get_window_size};

mod audio;
mod hotkey;
mod recognizer;
mod recorder;
mod tray;

fn main() -> eframe::Result {
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
    )
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
