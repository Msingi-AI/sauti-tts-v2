"""Prepare WAXAL Swahili TTS data into an LJSpeech-style layout.

Output:
  <out>/wavs/<id>.wav            mono wav at --sr (24 kHz default)
  <out>/metadata.csv             id|raw_text|normalized_text (pipe-separated)
  <out>/speakers.json            per-speaker clip counts and hours

The WaxalNLP schema is introspected at runtime (column names were not
verifiable at design time); the script fails loudly if it cannot find a
text column rather than guessing silently.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from swahili_text import normalize_for_tts

TEXT_CANDIDATES = ("text", "transcription", "transcript", "sentence", "normalized_text")
SPEAKER_CANDIDATES = ("speaker_id", "speaker", "voice_id", "voice", "reader_id")


def pick_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="google/WaxalNLP")
    ap.add_argument("--config", default="swa_tts")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--min-dur", type=float, default=1.0, help="seconds; shorter clips are dropped")
    ap.add_argument("--max-dur", type=float, default=15.0, help="seconds; longer clips are dropped")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split)
    print(f"Loaded {args.dataset}/{args.config}[{args.split}]: {len(ds)} rows")
    print(f"Columns: {ds.column_names}")

    text_col = pick_column(ds.column_names, TEXT_CANDIDATES)
    if text_col is None:
        sys.exit(f"No text column found among {TEXT_CANDIDATES}; inspect the columns above.")
    speaker_col = pick_column(ds.column_names, SPEAKER_CANDIDATES)
    if speaker_col is None:
        print("WARNING: no speaker column found; speakers.json will use a single bucket.")

    ds = ds.cast_column("audio", Audio(sampling_rate=args.sr))

    wav_dir = args.out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    speakers = defaultdict(lambda: {"clips": 0, "seconds": 0.0})
    rows, dropped = [], 0

    for i, ex in enumerate(tqdm(ds, desc="exporting")):
        audio = ex["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if not (args.min_dur <= duration <= args.max_dur):
            dropped += 1
            continue
        raw = ex[text_col].strip()
        if not raw:
            dropped += 1
            continue
        speaker = str(ex[speaker_col]) if speaker_col else "unknown"
        clip_id = f"waxal_sw_{i:06d}"
        sf.write(wav_dir / f"{clip_id}.wav", audio["array"], args.sr)
        rows.append(f"{clip_id}|{raw}|{normalize_for_tts(raw)}")
        speakers[speaker]["clips"] += 1
        speakers[speaker]["seconds"] += duration

    (args.out / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    stats = {
        spk: {"clips": s["clips"], "hours": round(s["seconds"] / 3600, 2)}
        for spk, s in sorted(speakers.items())
    }
    (args.out / "speakers.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    total_hours = sum(s["seconds"] for s in speakers.values()) / 3600
    print(f"Wrote {len(rows)} clips ({total_hours:.1f} h), dropped {dropped} "
          f"(outside {args.min_dur}-{args.max_dur}s or empty text)")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
