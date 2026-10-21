//! `ming parity`: each stage, teacher-forced, against the reference's goldens.
//!
//! Each stage gets the reference's own inputs (prompt ids, `z`, history, noise, latents), so
//! errors do not compound through the autoregressive loop. Reported as relative L2
//! (`‖ours − ref‖ / ‖ref‖`), worst case over frames.

use std::path::Path;

use anyhow::{Result, ensure};
use ming::flow_head::LATENT;
use ming::model::HIDDEN;
use ming::pipeline::Stage;
use ming::preset::Preset;
use ming::prompt::{Prompt, Voice};
use ming::session::Engine;

use compile::Checkpoint;

const SENTENCE_A: &str = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";

/// The reference's tensors for sentence A: per frame `z`, the aggregator's output (`agg`), the
/// flow head's history, noise and latent, and the stop logits; the whole waveform.
struct Goldens {
    ids: Vec<u32>,
    z: Vec<f32>,
    agg: Vec<f32>,
    hist: Vec<f32>,
    noise: Vec<f32>,
    latent: Vec<f32>,
    stop: Vec<f32>,
    wav: Vec<f32>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let ids = file.i32("ids")?;
        Ok(Self {
            ids: ids.into_iter().map(|id| id as u32).collect(),
            z: file.f32("z")?,
            agg: file.f32("agg")?,
            hist: file.f32("hist")?,
            noise: file.f32("noise")?,
            latent: file.f32("latent")?,
            stop: file.f32("stop")?,
            wav: file.f32("wav")?,
        })
    }

    fn frames(&self) -> usize {
        self.latent.len() / LATENT
    }

    fn z(&self, frame: usize) -> &[f32] {
        &self.z[frame * HIDDEN..(frame + 1) * HIDDEN]
    }

    fn latent(&self, frame: usize) -> &[f32] {
        &self.latent[frame * LATENT..(frame + 1) * LATENT]
    }

    fn aggregated(&self) -> impl Iterator<Item = &[f32]> {
        self.agg.as_chunks::<HIDDEN>().0.iter().map(|row| &row[..])
    }
}

fn relative_l2(ours: &[f32], reference: &[f32]) -> f64 {
    let diff: f64 = ours
        .iter()
        .zip(reference)
        .map(|(value, want)| f64::from(value - want).powi(2))
        .sum();
    let norm: f64 = reference.iter().map(|want| f64::from(*want).powi(2)).sum();
    (diff / norm).sqrt()
}

pub fn run(engine: &mut Engine, goldens: &Path) -> Result<()> {
    let goldens = Goldens::read(goldens)?;
    let prompt = engine
        .prompter()
        .build(SENTENCE_A, &Voice::Builtin("US_James".into()), None)?;
    ensure!(
        prompt.ids == goldens.ids,
        "prompt ids differ:\n ours {:?}\n ref  {:?}",
        prompt.ids,
        goldens.ids
    );
    println!("prompt ids: {} tokens, identical", goldens.ids.len());
    for stage in Stage::ALL {
        let report = match stage {
            Stage::Prefill => prefill(engine, &goldens, &prompt),
            Stage::Flow => flow(engine, &goldens),
            Stage::Vocode => vocode(engine, &goldens)?,
            Stage::Aggregate => aggregate(engine, &goldens),
            Stage::Decode => decode(engine, &goldens),
        };
        println!("{report}");
    }
    Ok(())
}

fn prefill(engine: &mut Engine, goldens: &Goldens, prompt: &Prompt) -> String {
    let z = engine.prefill(prompt);
    format!("z after prefill: rel {:.2e}", relative_l2(&z, goldens.z(0)))
}

fn flow(engine: &mut Engine, goldens: &Goldens) -> String {
    let frames = goldens.frames();
    let mut worst = 0f64;
    let mut decisions = 0;
    for frame in 0..frames {
        let range = frame * LATENT..(frame + 1) * LATENT;
        let (history, noise) = (&goldens.hist[range.clone()], &goldens.noise[range]);
        let ours = engine.sample(goldens.z(frame), history, noise, Preset::Quality);
        worst = worst.max(relative_l2(&ours, goldens.latent(frame)));
        let [keep, end] = engine.stop_logits(goldens.z(frame));
        let reference = &goldens.stop[frame * 2..frame * 2 + 2];
        decisions += usize::from((end > keep) == (reference[1] > reference[0]));
    }
    format!(
        "flow head latent: worst rel {worst:.2e}; stop decisions agree on {decisions}/{frames} frames"
    )
}

fn vocode(engine: &mut Engine, goldens: &Goldens) -> Result<String> {
    let reference = &goldens.wav;
    let ours = engine.vocode(&goldens.latent);
    ensure!(
        ours.len() == reference.len(),
        "vocoder: {} samples, reference {}",
        ours.len(),
        reference.len()
    );
    let max_abs = ours
        .iter()
        .zip(reference)
        .map(|(value, want)| (value - want).abs())
        .fold(0f32, f32::max);
    Ok(format!(
        "vocoder: {} samples, rel {:.2e}, max abs {max_abs:.2e}",
        ours.len(),
        relative_l2(&ours, reference)
    ))
}

fn aggregate(engine: &mut Engine, goldens: &Goldens) -> String {
    let mut worst = 0f64;
    for (frame, embedding) in goldens.aggregated().enumerate() {
        let ours = engine.aggregate(goldens.latent(frame));
        worst = worst.max(relative_l2(&ours, embedding));
    }
    format!("aggregator: worst rel {worst:.2e}")
}

/// The LLM at frame `k + 1`, fed the reference's aggregated embedding of frame `k`. Runs after
/// prefill, whose KV cache it extends.
fn decode(engine: &mut Engine, goldens: &Goldens) -> String {
    let mut worst = 0f64;
    let first = goldens.ids.len();
    for (frame, embedding) in goldens.aggregated().enumerate() {
        let ours = engine.decode(embedding, first + frame);
        worst = worst.max(relative_l2(&ours, goldens.z(frame + 1)));
    }
    format!(
        "z per frame (LLM decode, teacher-forced): worst rel {worst:.2e} over {} frames",
        goldens.frames() - 1
    )
}
