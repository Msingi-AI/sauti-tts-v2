"""Synthesize the eval set with Chatterbox Multilingual zero-shot Swahili.

Requires `pip install chatterbox-tts` (kept out of requirements.txt because it
pins its own dependency stack). The import path and generate() signature follow
the resemble-ai/chatterbox README — if they drift, check the pinned version's
README first.

Pass --ref-wav to clone a specific voice (e.g. a WAXAL speaker); without it the
model uses its built-in default voice.
"""

import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from swahili_text import normalize_for_tts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sentences", type=Path, default=Path("data/eval/eval_sentences.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref-wav", type=Path, default=None, help="reference audio for voice cloning")
    ap.add_argument("--language-id", default="sw")
    args = ap.parse_args()

    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError:
        raise SystemExit("chatterbox-tts is not installed: pip install chatterbox-tts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.sentences, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="synthesizing"):
        kwargs = {"language_id": args.language_id}
        if args.ref_wav:
            kwargs["audio_prompt_path"] = str(args.ref_wav)
        wav = model.generate(normalize_for_tts(row["text"]), **kwargs)
        sf.write(args.out / f"{row['id']}.wav", wav.squeeze().cpu().numpy(), model.sr)

    print(f"Wrote {len(rows)} wavs to {args.out} at {model.sr} Hz")


if __name__ == "__main__":
    main()
