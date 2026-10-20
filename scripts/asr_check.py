"""Intelligibility check: transcribe WAVs with faster-whisper (small, int8) and score the word
error rate against the text that was synthesized.

Usage (any venv with faster-whisper, e.g. ming-05b-experiment/.venv-asr):
  python scripts/asr_check.py --text "The evening wind ..." out/a.wav [out/b.wav ...]
  python scripts/asr_check.py --max-wer 0.1 --text "..." out/a.wav     # exit 1 above the bound
"""
import argparse
import re
import sys

# Spoken forms of the numerals in the test sentences, so "1998" and "nineteen ninety-eight"
# score the same.
SPOKEN = {"1998": "nineteen ninety eight", "2,500": "twenty five hundred", "2500": "twenty five hundred",
          "40%": "forty percent", "3:00": "three oclock", "3 o'clock": "three oclock",
          "three o'clock": "three oclock"}


def words(text):
    text = text.lower()
    for written, spoken in SPOKEN.items():
        text = text.replace(written, spoken)
    text = text.replace("'", "").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).split()


def wer(reference, hypothesis):
    r, h = words(reference), words(hypothesis)
    row = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        diagonal, row[0] = row[0], i
        for j in range(1, len(h) + 1):
            above = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, diagonal + (r[i - 1] != h[j - 1]))
            diagonal = above
    return row[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-wer", type=float)
    ap.add_argument("--language", default="en")
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    worst = 0.0
    for path in args.wavs:
        segments, _ = model.transcribe(path, beam_size=5, language=args.language)
        hypothesis = " ".join(s.text.strip() for s in segments)
        score = wer(args.text, hypothesis)
        worst = max(worst, score)
        print(f"WER={score:.3f} {path} | {hypothesis}")
    if args.max_wer is not None and worst > args.max_wer:
        sys.exit(f"WER {worst:.3f} > {args.max_wer}")


if __name__ == "__main__":
    main()
