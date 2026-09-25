use anyhow::Result;

use crate::recognizer::{ReadyHook, Recognizer};
use crate::recorder::{InputDevice, Recorder};

pub enum PipelineEvent {
    Ready,
    Recognized(String),
    Failed(anyhow::Error),
}

pub struct Pipeline {
    recorder: Recorder,
    recognizer: Recognizer,
    ready_hook: ReadyHook,
    loading: bool,
    active: bool,
}

impl Pipeline {
    pub fn new() -> Result<Self> {
        let (recognizer, ready_hook) = Recognizer::load()?;
        let recorder = Recorder::new()?;

        Ok(Self {
            recorder,
            recognizer,
            ready_hook,
            loading: true,
            active: false,
        })
    }

    pub fn poll(&mut self) -> Vec<PipelineEvent> {
        let mut events = Vec::new();

        if self.loading {
            match self.ready_hook.poll() {
                Ok(true) => {
                    self.loading = false;
                    events.push(PipelineEvent::Ready);
                }
                Ok(false) => {}
                Err(err) => {
                    self.loading = false;
                    events.push(PipelineEvent::Failed(err));
                }
            }
        }

        let samples = self.recorder.poll();
        if self.active && !samples.is_empty() {
            self.recognizer.push(samples, self.recorder.sample_rate());
        }

        while let Some(text) = self.recognizer.poll() {
            events.push(PipelineEvent::Recognized(text));
        }

        events
    }

    pub fn start(&mut self) {
        if self.loading || self.active {
            return;
        }
        if self.recorder.start().is_ok() {
            self.recognizer.begin();
            self.active = true;
        }
    }

    pub fn stop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;

        if let Ok(samples) = self.recorder.stop() {
            if !samples.is_empty() {
                self.recognizer.push(samples, self.recorder.sample_rate());
            }
        }
        self.recognizer.end();
    }

    pub fn dbfs(&mut self) -> f32 {
        self.recorder.dbfs()
    }

    pub fn list_devices(&self) -> Vec<InputDevice> {
        self.recorder.list_devices()
    }

    pub fn current_device(&self) -> &str {
        self.recorder.current_device()
    }

    pub fn select_device(&mut self, id: &str) -> Result<()> {
        self.recorder.select_device(id)
    }
}
