"""Dump golden tensors from the fp32 PyTorch reference for the Rust parity tests.

Runs Ming-3.0-tts-0.5B (task IP, voice US_James, zero speaker embedding) on one sentence and
records every intermediate the runtime reproduces: the prompt ids, the input embeddings, the
LLM output of layer 0 and the conditioning vector `z` per frame, the flow head's noise and
output latent per frame, the stop logits, the aggregator output and the VAE waveform.

The flow head's noise is drawn here (seeded per frame) and recorded, so the Rust tests inject
the same noise instead of reproducing torch's RNG.

`--reference` is a directory holding a checkout of github.com/inclusionAI/Ming-3.0-tts
(`Ming-3.0-tts/`, commit e999c39) and the checkpoint (`Ming-3.0-tts-0.5b/`); run it with a
Python environment that has that repository's requirements.

Usage:
  python scripts/dump_goldens.py --reference ../ming-05b-experiment --out fixtures/goldens/ref_f32_A.safetensors
"""
import argparse
import os
import sys

SENTENCE_A = "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold."
SEED = 1895


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="dir with Ming-3.0-tts/ and Ming-3.0-tts-0.5b/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=SENTENCE_A)
    ap.add_argument("--voice", default="US_James")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    reference = os.path.abspath(args.reference)
    repo = os.path.join(reference, "Ming-3.0-tts")
    sys.path.insert(0, repo)
    os.chdir(repo)

    import torch
    import tts_tasks
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from ming_tts import SPEAKER_EMBEDDING_DIM, trim_trailing_silence
    from modeling_bailingmm import BailingMMNativeForConditionalGeneration

    # Built like `MingTTS.__init__`, but in f32 (MingTTS casts every buffer to bf16, which also
    # rounds the RoPE frequencies and the iSTFT window). The reference's
    # `autocast(device_type="cuda")` does not touch CPU tensors, so everything runs in f32.
    checkpoint = os.path.join(reference, "Ming-3.0-tts-0.5b")
    model = BailingMMNativeForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    model.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    sample_rate = model.config.audio_tokenizer_config.sample_rate
    rec = {"z": [], "noise": [], "hist": [], "latent": [], "stop": [], "agg": []}

    flow = model.flowloss
    frame = {"n": 0}

    def sample(z, latent_history, cfg=1.0, patch_size=1, sigma=0.25, temperature=0, steps=10):
        torch.manual_seed(SEED + frame["n"])
        noise = torch.randn((1, 1, flow.z_channels), dtype=z.dtype)
        sde_noise = torch.zeros((args.steps, 1, 1, flow.z_channels), dtype=z.dtype)
        y = flow.cfm.sample(noise=noise, c=z, latent_history=latent_history, sde_noise=sde_noise,
                            cfg_scale=cfg, patch_size=patch_size, sigma=sigma, temperature=temperature)
        for key, value in (("z", z), ("noise", noise), ("hist", latent_history), ("latent", y)):
            rec[key].append(value.detach().reshape(-1).clone())
        frame["n"] += 1
        return y

    flow.sample = sample
    # A forward hook that returns a value replaces the module's output: these return None.
    def record(key, value):
        rec[key].append(value.detach().reshape(-1).clone())

    def record_first(key, value):
        first.setdefault(key, value.detach().clone())

    model.stop_head.register_forward_hook(lambda m, i, o: record("stop", o))
    model.linear_proj_audio.register_forward_hook(lambda m, i, o: record("agg", o))

    first = {}
    llm = model.model.model
    llm.embed_tokens.register_forward_hook(lambda m, i, o: record_first("ids", i[0]))
    llm.layers[0].register_forward_hook(lambda m, i, o: record_first("layer0", o[0]))
    orig_forward = model.model.forward

    def llm_forward(*a, **k):
        if "embeds" not in first:
            first["embeds"] = k["inputs_embeds"].detach().clone()
        return orig_forward(*a, **k)

    model.model.forward = llm_forward

    decoded = {}
    orig_decode = model.audio.decode

    def decode(latent, **k):
        result = orig_decode(latent, **k)
        decoded["latents"] = latent.detach().clone()
        decoded["wav"] = result[0].detach().reshape(-1).clone()
        return result

    model.audio.decode = decode

    # `MingTTS.synthesize(task="IP", raw_text=True, use_zero_spk_emb=True)` for one fragment.
    request = tts_tasks.build_task_generation_kwargs(
        "IP", text=args.text, controls={"ip": args.voice}, use_zero_spk_emb=True)
    with torch.inference_mode():
        wav = model.generate(
            prompt=request["prompt"], text=request["text"],
            spk_emb=[torch.zeros(1, SPEAKER_EMBEDDING_DIM)], instruction=request["instruction"],
            cfg=request["cfg"], sigma=request["sigma"], temperature=request["temperature"],
            use_zero_spk_emb=True)
        trimmed = trim_trailing_silence(wav, sample_rate)

    stack = lambda key: torch.stack(rec[key]).float().contiguous()
    tensors = {
        "ids": first["ids"].reshape(-1).to(torch.int32),
        "embeds": first["embeds"][0].float().contiguous(),
        "layer0": first["layer0"][0].float().contiguous(),
        "z": stack("z"),
        "noise": stack("noise"),
        "hist": stack("hist"),
        "latent": stack("latent"),
        "stop": stack("stop"),
        "agg": stack("agg"),
        "wav": decoded["wav"].float().contiguous(),
        "wav_trimmed": trimmed.reshape(-1).float().contiguous(),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(tensors, out, metadata={"text": args.text, "voice": args.voice, "steps": str(args.steps)})
    for name, t in tensors.items():
        print(f"{name:12s} {tuple(t.shape)}")
    print(f"frames={frame['n']} audio={trimmed.numel() / sample_rate:.2f}s -> {out}")


if __name__ == "__main__":
    main()
