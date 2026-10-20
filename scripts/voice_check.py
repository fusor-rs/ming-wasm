"""Voice similarity: cosine of CAM++ speaker embeddings (the checkpoint's campplus.onnx, the
extractor the model was trained with) between a reference WAV and candidates.

Usage (reference venv: onnxruntime + torchaudio):
  python scripts/voice_check.py --campplus <dir>/campplus.onnx ref.wav a.wav b.wav
"""
import argparse

import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


def embedding(session, path):
    wav, rate = torchaudio.load(path)
    wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), rate, 16000)
    feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    out = session.run(None, {session.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0]
    return torch.tensor(out).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campplus", required=True)
    ap.add_argument("reference")
    ap.add_argument("candidates", nargs="+")
    args = ap.parse_args()
    session = onnxruntime.InferenceSession(args.campplus, providers=["CPUExecutionProvider"])
    ref = embedding(session, args.reference)
    for path in args.candidates:
        sim = torch.nn.functional.cosine_similarity(ref, embedding(session, path), dim=0).item()
        print(f"similarity={sim:.3f} {path}")


if __name__ == "__main__":
    main()
