use fusor::{JsInputs, Signal, signal};
use serde::Deserialize;

const SAMPLE_TEXT: &str = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";
/// The voice value that asks for a voice described in words.
const DESIGN: &str = "design";

#[derive(Clone, PartialEq, Deserialize)]
struct Voice {
    id: String,
    label: String,
}

/// What `web/app.js` reports, one JSON object per `engine` event.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Report {
    Loading {
        fraction: f64,
        detail: String,
    },
    Ready {
        voices: Vec<Voice>,
        runtime: String,
    },
    Progress {
        fraction: f64,
        status: String,
    },
    Done {
        status: String,
        audio: Option<String>,
        stats: String,
    },
    /// A synthesis failed; the engine can run the next one.
    Error {
        message: String,
    },
    /// The model could not load, or the engine died: nothing more can run in this page.
    Broken {
        message: String,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Loading,
    Ready,
    Generating,
    Broken,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Generating => "generating",
            Self::Broken => "broken",
        }
    }
}

#[derive(JsInputs)]
struct App {
    #[js]
    text: Signal<String>,
    #[js]
    voice: Signal<String>,
    #[js]
    description: Signal<String>,
    #[js]
    preset: Signal<String>,
    /// Generate and Stop count their clicks; `web/app.js` acts on every new count.
    #[js]
    generations: Signal<u32>,
    #[js]
    stops: Signal<u32>,
    phase: Signal<Phase>,
    fraction: Signal<f64>,
    detail: Signal<String>,
    voices: Signal<Vec<Voice>>,
    status: Signal<String>,
    audio: Signal<Option<String>>,
    stats: Signal<String>,
    runtime: Signal<String>,
}

impl App {
    fn new() -> Self {
        Self {
            text: signal(SAMPLE_TEXT.into()),
            voice: signal("US_James".into()),
            description: signal(String::new()),
            preset: signal("quality".into()),
            generations: signal(0),
            stops: signal(0),
            phase: signal(Phase::Loading),
            fraction: signal(0.0),
            detail: signal("Starting…".into()),
            voices: signal(Vec::new()),
            status: signal(String::new()),
            audio: signal(None),
            stats: signal(String::new()),
            runtime: signal(String::new()),
        }
    }

    fn generate(&self) {
        if self.text.get().trim().is_empty() {
            self.status.set("Type some text first.".into());
            return;
        }
        if self.describing() && self.description.get().trim().is_empty() {
            self.status.set("Describe the voice first.".into());
            return;
        }
        self.phase.set(Phase::Generating);
        self.fraction.set(0.0);
        self.status.set("Preparing…".into());
        self.generations.update(|count| *count += 1);
    }

    fn stop(&self) {
        self.stops.update(|count| *count += 1);
    }

    fn receive(&self, event: web_sys::Event) {
        let json = fusor::js::event_detail::<String>(&event).expect("web/app.js sends strings");
        let report = serde_json::from_str(&json).expect("web/app.js sends engine reports");
        match report {
            Report::Loading { fraction, detail } => {
                self.fraction.set(fraction);
                self.detail.set(detail);
            }
            Report::Ready {
                mut voices,
                runtime,
            } => {
                voices.push(Voice {
                    id: DESIGN.into(),
                    label: "Describe a voice…".into(),
                });
                self.voices.set(voices);
                self.runtime.set(runtime);
                self.phase.set(Phase::Ready);
            }
            Report::Progress { fraction, status } => {
                self.fraction.set(fraction);
                self.status.set(status);
            }
            Report::Done {
                status,
                audio,
                stats,
            } => {
                self.status.set(status);
                self.audio.set(audio);
                self.stats.set(stats);
                self.phase.set(Phase::Ready);
            }
            Report::Error { message } => {
                self.status.set(format!("error: {message}"));
                self.phase.set(Phase::Ready);
            }
            Report::Broken { message } => {
                self.detail.set(message);
                self.phase.set(Phase::Broken);
            }
        }
    }

    fn describing(&self) -> bool {
        self.voice.get() == DESIGN
    }
}

fusor::template!("web/index.html");
