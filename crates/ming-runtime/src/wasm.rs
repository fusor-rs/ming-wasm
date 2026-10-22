use ming::prompt::VOICES;
use ming::session::{Engine, Request};
use runtime::{error, json};
use wasm_bindgen::prelude::*;

/// `[{"id","label"}]` of the built-in voices.
#[wasm_bindgen]
pub fn voices() -> String {
    let list: Vec<_> = VOICES
        .iter()
        .map(|(id, label)| serde_json::json!({ "id": id, "label": label }))
        .collect();
    serde_json::Value::from(list).to_string()
}

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts a synthesis (`session::Request` as JSON); returns the plan (`{"fragments"}`).
    pub fn start(&mut self, request_json: &str) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request_json)
            .map_err(|error| JsError::new(&format!("request: {error}")))?;
        Ok(json(&self.shell.engine()?.start(request).map_err(error)?))
    }

    /// One frame; returns `{"fragment","fragments","frame","done"}`.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// The 44.1 kHz samples produced since the last call.
    pub fn take_audio(&mut self) -> Result<Vec<f32>, JsError> {
        Ok(self.shell.engine()?.take_audio())
    }
}
