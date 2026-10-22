//! `ming`: the native host of the runtime.
//!
//! - `compile`: checkpoint → bundle (the `ming-compile` crate).
//! - `say`: text → WAV with a bundle, printing the real-time factor.
//! - `parity`: every component against the PyTorch goldens (`scripts/dump_goldens.py`).
//! - `bench`: milliseconds per frame of each component.

mod bench;
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ming::SAMPLE_RATE;
use ming::preset::Preset;
use ming::prompt::Voice;
use ming::session::{Engine, Request};

#[derive(Parser)]
#[command(name = "ming", version)]
struct Cli {
    /// Compute threads (default: min(8, cores − 2)).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the checkpoint directory (model.safetensors, tokenizer.json) into a bundle.
    Compile {
        checkpoint: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "mix")]
        profile: ming_compile::Profile,
    },
    /// Synthesize `text` into a WAV file.
    Say {
        bundle: PathBuf,
        text: String,
        #[arg(short, long, default_value = "out.wav")]
        out: PathBuf,
        /// A built-in voice (`US_James`, `US_Emma`, `UK_George`, `灵小柔`, …).
        #[arg(long, default_value = "US_James")]
        voice: String,
        /// Describe the voice instead of using a built-in one.
        #[arg(long, conflicts_with = "freeform")]
        design: Option<String>,
        /// A free-form instruction for voice and delivery.
        #[arg(long)]
        freeform: Option<String>,
        #[arg(long, value_enum, default_value = "quality")]
        preset: PresetArg,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Compare every component with the golden tensors of the reference.
    Parity { bundle: PathBuf, goldens: PathBuf },
    /// Time each component of the per-frame loop.
    Bench { bundle: PathBuf },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum PresetArg {
    Quality,
    Balanced,
    Fast,
}

impl From<PresetArg> for Preset {
    fn from(preset: PresetArg) -> Self {
        match preset {
            PresetArg::Quality => Preset::Quality,
            PresetArg::Balanced => Preset::Balanced,
            PresetArg::Fast => Preset::Fast,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    harness::threads(cli.threads)?;
    match cli.command {
        Command::Compile {
            checkpoint,
            out,
            profile,
        } => harness::compile(|| ming_compile::run(&checkpoint, &out, profile))?,
        Command::Say {
            bundle,
            text,
            out,
            voice,
            design,
            freeform,
            preset,
            seed,
        } => {
            let mut engine = harness::load(&bundle)?;
            let voice = match (design, freeform) {
                (Some(style), _) => Voice::Design(style),
                (_, Some(instruction)) => Voice::Freeform(instruction),
                _ => Voice::Builtin(voice),
            };
            let request = Request {
                text,
                voice,
                preset: preset.into(),
                seed,
                language: None,
                max_frames: None,
            };
            say(&mut engine, request, &out)?;
        }
        Command::Parity { bundle, goldens } => parity::run(&mut harness::load(&bundle)?, &goldens)?,
        Command::Bench { bundle } => bench::run(&mut harness::load(&bundle)?)?,
    }
    Ok(())
}

fn say(engine: &mut Engine, request: Request, out: &Path) -> Result<()> {
    let start = Instant::now();
    let plan = engine.start(request)?;
    let mut audio = Vec::new();
    let mut frames = 0;
    let mut first_audio = None;
    loop {
        let progress = engine.step()?;
        frames += 1;
        audio.extend(engine.take_audio());
        if first_audio.is_none() && !audio.is_empty() {
            first_audio = Some(start.elapsed().as_secs_f64());
        }
        if progress.done {
            break;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let seconds = audio.len() as f64 / f64::from(SAMPLE_RATE);
    std::fs::write(out, ming::wav::encode(&audio, SAMPLE_RATE))?;
    println!(
        "{} fragment(s), {frames} frames, {seconds:.2}s audio in {elapsed:.2}s: RTF {:.3}, {:.1} ms/frame, first audio {:.2}s -> {}",
        plan.fragments.len(),
        elapsed / seconds,
        elapsed * 1000.0 / frames as f64,
        first_audio.unwrap_or(elapsed),
        out.display()
    );
    Ok(())
}
