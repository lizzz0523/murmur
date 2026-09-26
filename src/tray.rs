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
const HISTORY_PREFIX: &str = "history:";
const HISTORY_LABEL_MAX: usize = 40;

pub enum TrayAction {
    SelectDevice(String),
    CopyText(String),
    RefreshDevices,
    Quit,
}

pub struct Tray {
    _icon: TrayIcon,
    menu: Menu,
    device_submenu: Submenu,
    device_items: RefCell<Vec<(String, CheckMenuItem)>>,
    history_items: RefCell<Vec<MenuItem>>,
    history_decorations: RefCell<Vec<PredefinedMenuItem>>,
    actions: Arc<Mutex<VecDeque<TrayAction>>>,
}

impl Tray {
    pub fn new(ctx: &egui::Context) -> anyhow::Result<Self> {
        let refresh = MenuItem::new("刷新设备", true, None);
        let refresh_id = refresh.id().clone();

        let quit = MenuItem::new("退出", true, None);
        let quit_id = quit.id().clone();

        let device_submenu = Submenu::new("麦克风", true);

        let menu = Menu::new();
        menu.append(&device_submenu)?;
        menu.append(&refresh)?;
        menu.append(&quit)?;

        let tray = TrayIconBuilder::new()
            .with_icon(load_icon()?)
            .with_icon_as_template(true)
            .with_tooltip("murmur")
            .with_menu(Box::new(menu.clone()))
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
            } else if let Some(text) = raw.strip_prefix(HISTORY_PREFIX) {
                TrayAction::CopyText(text.to_string())
            } else {
                return;
            };
            menu_actions.lock().unwrap().push_back(action);
            menu_ctx.request_repaint();
        }));

        Ok(Self {
            _icon: tray,
            menu,
            device_submenu,
            device_items: RefCell::new(Vec::new()),
            history_items: RefCell::new(Vec::new()),
            history_decorations: RefCell::new(Vec::new()),
            actions,
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
            let _ = self.device_submenu.remove(item);
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
            if self.device_submenu.append(&item).is_ok() {
                items.push((device.id.clone(), item));
            }
        }
    }

    pub fn set_history(&self, history: &VecDeque<String>) {
        const BASE: usize = 2;

        for decoration in self.history_decorations.borrow().iter() {
            let _ = self.menu.remove(decoration);
        }
        for item in self.history_items.borrow().iter() {
            let _ = self.menu.remove(item);
        }
        self.history_decorations.borrow_mut().clear();
        self.history_items.borrow_mut().clear();

        let mut decorations = self.history_decorations.borrow_mut();
        let mut insert_at = BASE;

        let leading = PredefinedMenuItem::separator();
        if self.menu.insert(&leading, insert_at).is_ok() {
            decorations.push(leading);
        }
        insert_at += 1;

        let header = PredefinedMenuItem::section_header("最近记录");
        if self.menu.insert(&header, insert_at).is_ok() {
            decorations.push(header);
        }
        insert_at += 1;

        let mut items = self.history_items.borrow_mut();
        if history.is_empty() {
            let placeholder = MenuItem::new("（暂无记录）", false, None);
            if self.menu.insert(&placeholder, insert_at).is_ok() {
                items.push(placeholder);
            }
            insert_at += 1;
        } else {
            for text in history {
                let item = MenuItem::with_id(
                    MenuId::new(format!("{HISTORY_PREFIX}{text}")),
                    truncate(text, HISTORY_LABEL_MAX),
                    true,
                    None,
                );
                if self.menu.insert(&item, insert_at).is_ok() {
                    items.push(item);
                }
                insert_at += 1;
            }
        }

        let trailing = PredefinedMenuItem::separator();
        if self.menu.insert(&trailing, insert_at).is_ok() {
            decorations.push(trailing);
        }
    }

    pub fn poll(&self) -> Option<TrayAction> {
        self.actions.lock().unwrap().pop_front()
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
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
