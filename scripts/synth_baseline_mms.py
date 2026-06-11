"""Synthesize the eval set with facebook/mms-tts-swh (VITS) — the baseline floor.

Writes <out>/<id>.wav for every row in the eval CSV.
"""

import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, VitsModel

from swahili_text import normalize_for_tts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="facebook/mms-tts-swh")
    ap.add_argument("--sentences", type=Path, default=Path("data/eval/eval_sentences.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VitsModel.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.sentences, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="synthesizing"):
        inputs = tokenizer(normalize_for_tts(row["text"]), return_tensors="pt").to(device)
        with torch.no_grad():
            wav = model(**inputs).waveform[0].cpu().numpy()
        sf.write(args.out / f"{row['id']}.wav", wav, model.config.sampling_rate)

    print(f"Wrote {len(rows)} wavs to {args.out} at {model.config.sampling_rate} Hz")


if __name__ == "__main__":
    main()
