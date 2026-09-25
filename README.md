# murmur

macOS 上的本地语音输入工具：按住快捷键说话，松开后自动识别、润色并粘贴到当前光标处。所有推理都在本机完成，音频不会离开设备。

## 功能特性

- **按住说话**：按住右侧 `Ctrl`（或 `Fn`）录音，松开即开始识别。
- **悬浮胶囊**：屏幕底部居中的置顶胶囊实时显示状态（Ready / 录音波形 / Thinking...）。
- **自动粘贴**：识别并润色后的文本通过模拟键盘输入写入当前焦点。
- **菜单栏图标**：可列出并切换输入设备、刷新设备列表、退出应用。
- **全本地推理**：语音降噪、识别与文本润色均在本地运行。

## 处理流程

```
录音 → 重采样到 16kHz → 高通滤波 → GTCRN 降噪 → 响度归一化 → 峰值限制
    → Silero VAD 分段 → Qwen3-ASR 识别 → Qwen3-4B 润色
    → 粘贴
```

## 系统要求

- macOS
- Rust 1.90 或更高（edition 2024）
- 首次运行需要联网下载模型（之后缓存复用）
- 从源码构建需要 `cmake` 与 `clang`（`llama-cpp-sys` 会用 cmake + bindgen 编译 llama.cpp，`libsamplerate-sys` 也会用 cmake 编译 libsamplerate）；若 bindgen 找不到 `libclang`，设置 `LIBCLANG_PATH` 指向包含 `libclang.dylib` 的目录。
- CMake 4 及以上版本已由仓库内 `.cargo/config.toml` 的 `CMAKE_POLICY_VERSION_MINIMUM=3.5` 自动处理，无需手动设置。

## 构建与运行

```bash
cargo build --release
```

首次构建会现场编译 llama.cpp，耗时较长；首次启动会通过 Hugging Face 下载所需模型，耗时取决于网络。国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com` 走镜像。

首次使用需在「系统设置 → 隐私与安全性」中授权：

- **麦克风**：录音。
- **输入监控**：`handy-keys` 监听全局快捷键。
- **辅助功能**：`enigo` 模拟键盘输入以粘贴文本。

## 使用说明

- 按住右侧 `Ctrl` 或 `Fn` 开始录音，松开结束。
- 状态依次为 `Ready`（就绪）、录音波形（录音中）、`Thinking...`（识别/润色中）。
- 菜单栏图标菜单：
  - **麦克风**：列出所有输入设备，勾选当前设备，点击可切换（录音/识别过程中切换会被忽略）。
  - **刷新设备**：设备热插拔后手动刷新列表。
  - **退出**：退出应用。
- 窗口为无边框、置顶、可拖动且不可缩放的悬浮胶囊，可拖到任意位置。

## 模型

| 用途 | 模型仓库 |
| --- | --- |
| 语音降噪 | `csukuangfj/speech-enhancement-models` |
| 语音识别 | `solavr/sherpa-onnx-qwen3-asr-1.7B-int8` |
| 语音活动检测 | `csukuangfj/vad` |
| 文本润色 | `unsloth/Qwen3-4B-GGUF` |

模型下载后由 `hf-hub` 缓存在本地。

## 主要依赖

- [`eframe`](https://crates.io/crates/eframe) / `egui`：悬浮窗口与界面绘制
- [`cpal`](https://crates.io/crates/cpal)：音频采集
- [`samplerate`](https://crates.io/crates/samplerate)：高质量抗混叠重采样（libsamplerate 绑定）
- [`sherpa-onnx`](https://crates.io/crates/sherpa-onnx)：降噪、VAD 与 ASR
- [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2)：文本润色（GGUF + Metal 加速）
- [`handy-keys`](https://crates.io/crates/handy-keys) / [`enigo`](https://crates.io/crates/enigo)：全局快捷键与模拟输入
- [`tray-icon`](https://crates.io/crates/tray-icon)：菜单栏图标与菜单

## 项目结构

```
src/
├── app.rs         应用状态机与界面绘制
├── audio.rs       降混、重采样、高通滤波、归一化、峰值限制
├── hotkey.rs      全局快捷键监听
├── hub.rs         模型下载与进度（resolve_model）
├── main.rs        程序入口、窗口配置与图标
├── recognizer.rs  降噪、VAD 分段与 ASR
├── recorder.rs    音频录制、设备枚举与切换
├── refiner.rs     Qwen3 文本润色（llama.cpp 推理）
└── tray.rs        菜单栏图标与菜单
assets/            应用图标与托盘模板图标
```

## 许可证

[MIT](LICENSE)
