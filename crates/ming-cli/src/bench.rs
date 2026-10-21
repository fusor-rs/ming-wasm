//! `ming bench`: milliseconds per call of each stage on this machine (the AR loop's per-frame
//! work), and the frame total they add up to.

use std::time::Instant;

use anyhow::Result;
use ming::flow_head::LATENT;
use ming::pipeline::Stage;
use ming::preset::Preset;
use ming::prompt::Voice;
use ming::session::Engine;
use ming::vocoder::FRAME_SAMPLES;

const SENTENCE: &str = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";
const CALLS: usize = 20;

/// Milliseconds per call of `call`, after one warm-up call.
fn time(mut call: impl FnMut()) -> f64 {
    call();
    let start = Instant::now();
    for _ in 0..CALLS {
        call();
    }
    start.elapsed().as_secs_f64() * 1000.0 / CALLS as f64
}

/// One frame's inputs, from a real prefill.
struct Frame {
    z: Vec<f32>,
    history: Vec<f32>,
    noise: Vec<f32>,
    latent: Vec<f32>,
    embedding: Vec<f32>,
    position: usize,
}

pub fn run(engine: &mut Engine) -> Result<()> {
    let prompt = engine
        .prompter()
        .build(SENTENCE, &Voice::Builtin("US_James".into()), None)?;
    let start = Instant::now();
    let z = engine.prefill(&prompt);
    let prefill = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "threads {}: prefill {} tokens {prefill:.0} ms",
        engine.threads(),
        prompt.ids.len()
    );
    let noise: Vec<f32> = (0..LATENT)
        .map(|i| ((i * 7919 % 113) as f32 / 56.0) - 1.0)
        .collect();
    let history = vec![0.0; LATENT];
    let latent = engine.sample(&z, &history, &noise, Preset::Quality);
    let embedding = engine.aggregate(&latent);
    let mut frame = Frame {
        z,
        history,
        noise,
        latent,
        embedding,
        position: prompt.ids.len(),
    };
    let mut flows = Vec::new();
    let mut rest = 0.0;
    for stage in Stage::ALL {
        match stage {
            Stage::Prefill => {}
            Stage::Flow => flows = flow(engine, &frame),
            _ => rest += per_frame(engine, &mut frame, stage),
        }
    }
    let frame_ms = 1000.0 * FRAME_SAMPLES as f64 / 44_100.0;
    for (preset, ms) in &flows {
        let total = rest + ms;
        println!(
            "  frame ({:8}) {total:6.2} ms → RTF {:.3}",
            format!("{preset:?}").to_lowercase(),
            total / frame_ms
        );
    }
    Ok(())
}

/// The flow head per preset.
fn flow(engine: &mut Engine, frame: &Frame) -> Vec<(Preset, f64)> {
    Preset::ALL
        .into_iter()
        .map(|preset| {
            let ms = time(|| drop(engine.sample(&frame.z, &frame.history, &frame.noise, preset)));
            let name = format!("{preset:?}").to_lowercase();
            println!("  flow {name:9} {ms:6.2} ms/frame");
            (preset, ms)
        })
        .collect()
}

/// A stage other than prefill and flow, in ms per frame.
fn per_frame(engine: &mut Engine, frame: &mut Frame, stage: Stage) -> f64 {
    let ms = match stage {
        Stage::Vocode => {
            let latents = frame.latent.repeat(CALLS);
            let start = Instant::now();
            let samples = engine.vocode(&latents).len();
            debug_assert!(samples >= (CALLS - 1) * FRAME_SAMPLES);
            start.elapsed().as_secs_f64() * 1000.0 / CALLS as f64
        }
        Stage::Aggregate => time(|| drop(engine.aggregate(&frame.latent))),
        Stage::Decode => time(|| {
            engine.decode(&frame.embedding, frame.position);
            frame.position += 1;
        }),
        Stage::Prefill | Stage::Flow => unreachable!("timed by `run`"),
    };
    println!("  {:12} {ms:6.2} ms/frame", stage.name());
    ms
}
