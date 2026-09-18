use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use eframe::egui;
use tray_icon::menu::{
    CheckMenuItem, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu,
};
use tray_icon::{TrayIcon, TrayIconBuilder};

use crate::recorder::InputDevice;

const DEVICE_PREFIX: &str = "device:";

pub enum TrayAction {
    SelectDevice(String),
    RefreshDevices,
    Quit,
}

pub struct Tray {
    _icon: TrayIcon,
    actions: Arc<Mutex<VecDeque<TrayAction>>>,
    submenu: Submenu,
    device_items: RefCell<Vec<(String, CheckMenuItem)>>,
}

impl Tray {
    pub fn new(ctx: &egui::Context) -> anyhow::Result<Self> {
        let refresh = MenuItem::new("刷新设备", true, None);
        let refresh_id = refresh.id().clone();

        let quit = MenuItem::new("退出", true, None);
        let quit_id = quit.id().clone();

        let submenu = Submenu::new("麦克风", true);

        let menu = Menu::new();
        menu.append(&submenu)?;
        menu.append(&refresh)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit)?;

        let tray = TrayIconBuilder::new()
            .with_icon(load_icon()?)
            .with_icon_as_template(true)
            .with_tooltip("murmur")
            .with_menu(Box::new(menu))
            .build()?;

        let actions = Arc::new(Mutex::new(VecDeque::new()));

        let menu_actions = actions.clone();
        let menu_ctx = ctx.clone();
        MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
            let raw = event.id.0.as_str();
            let action = if raw == quit_id.0.as_str() {
                TrayAction::Quit
            } else if raw == refresh_id.0.as_str() {
                TrayAction::RefreshDevices
            } else if let Some(id) = raw.strip_prefix(DEVICE_PREFIX) {
                TrayAction::SelectDevice(id.to_string())
            } else {
                return;
            };
            menu_actions.lock().unwrap().push_back(action);
            menu_ctx.request_repaint();
        }));

        Ok(Self {
            _icon: tray,
            actions,
            submenu,
            device_items: RefCell::new(Vec::new()),
        })
    }

    pub fn set_selected_device(&self, id: &str) {
        for (device_id, item) in self.device_items.borrow().iter() {
            item.set_checked(device_id == id);
        }
    }

    pub fn set_devices(&self, devices: &[InputDevice], current_id: &str) {
        let mut items = self.device_items.borrow_mut();

        for (_, item) in items.iter() {
            let _ = self.submenu.remove(item);
        }
        items.clear();

        for device in devices {
            if device.id.is_empty() {
                continue;
            }
            let label = if device.is_default {
                format!("{} (默认)", device.name)
            } else {
                device.name.clone()
            };
            let item = CheckMenuItem::with_id(
                MenuId::new(format!("{DEVICE_PREFIX}{}", device.id)),
                label,
                true,
                device.id == current_id,
                None,
            );
            if self.submenu.append(&item).is_ok() {
                items.push((device.id.clone(), item));
            }
        }
    }

    pub fn poll(&self) -> Option<TrayAction> {
        self.actions.lock().unwrap().pop_front()
    }
}

fn load_icon() -> anyhow::Result<tray_icon::Icon> {
    let data = eframe::icon_data::from_png_bytes(include_bytes!("../assets/trayicon.png"))?;
    Ok(tray_icon::Icon::from_rgba(
        data.rgba,
        data.width,
        data.height,
    )?)
}
