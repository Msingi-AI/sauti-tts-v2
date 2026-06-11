"""Transcript-agreement filter: drop clips whose audio disagrees with the transcript.

Transcribes every clip in a prepared dataset (see prepare_waxal.py) with an
ASR judge and keeps rows whose character error rate against the normalized
transcript is at or below --max-cer.

Output (next to metadata.csv):
  metadata_filtered.csv   surviving rows, same pipe-separated format
  filter_report.json      per-clip CER, kept/dropped counts, worst offenders
"""

import argparse
import json
from pathlib import Path

import jiwer
from tqdm import tqdm

from asr_judge import WHISPER_ZERO_SHOT, load_judge, transcribe
from swahili_text import normalize_for_asr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="dir containing metadata.csv and wavs/")
    ap.add_argument("--judge", default=WHISPER_ZERO_SHOT)
    ap.add_argument("--max-cer", type=float, default=0.10)
    args = ap.parse_args()

    rows = [line.split("|") for line in
            (args.data / "metadata.csv").read_text(encoding="utf-8").splitlines() if line]
    judge = load_judge(args.judge)

    kept, results = [], []
    for clip_id, raw, normalized in tqdm(rows, desc="transcribing"):
        hyp = transcribe(judge, args.data / "wavs" / f"{clip_id}.wav")
        ref_n, hyp_n = normalize_for_asr(normalized), normalize_for_asr(hyp)
        cer = jiwer.cer(ref_n, hyp_n) if ref_n else 1.0
        results.append({"id": clip_id, "cer": round(cer, 4), "ref": ref_n, "hyp": hyp_n})
        if cer <= args.max_cer:
            kept.append(f"{clip_id}|{raw}|{normalized}")

    (args.data / "metadata_filtered.csv").write_text("\n".join(kept) + "\n", encoding="utf-8")
    results.sort(key=lambda r: -r["cer"])
    report = {
        "judge": args.judge,
        "max_cer": args.max_cer,
        "total": len(rows),
        "kept": len(kept),
        "dropped": len(rows) - len(kept),
        "mean_cer": round(sum(r["cer"] for r in results) / max(len(results), 1), 4),
        "clips": results,
    }
    (args.data / "filter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(f"Kept {len(kept)}/{len(rows)} clips (mean CER {report['mean_cer']:.3f}); "
          f"worst offenders are at the top of filter_report.json")


if __name__ == "__main__":
    main()
