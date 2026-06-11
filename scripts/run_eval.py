"""Score a directory of synthesized wavs against the eval sentences.

Metrics:
  - WER and CER per ASR judge (default: zero-shot whisper-large-v3 plus a
    Swahili-fine-tuned Whisper). CER is the headline number — Swahili's
    agglutinative morphology inflates WER.
  - UTMOS quality proxy (English-trained; relative ranking signal only).
  - Optional speaker similarity (ECAPA cosine) against --ref-wav-dir with
    matching <id>.wav files.

Writes <out>.json (per-clip detail) and <out>.md (summary tables, overall and
per category).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import jiwer
import torch
import torchaudio
from tqdm import tqdm

from asr_judge import WHISPER_SWAHILI, WHISPER_ZERO_SHOT, load_judge, transcribe
from swahili_text import normalize_for_asr


def load_wav_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def utmos_scores(wav_paths):
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    return {p.stem: float(predictor(load_wav_16k(p), 16000).item()) for p in tqdm(wav_paths, desc="UTMOS")}


def speaker_similarity(wav_paths, ref_dir: Path):
    from speechbrain.inference.speaker import EncoderClassifier
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def embed(path):
        return encoder.encode_batch(load_wav_16k(path)).squeeze()

    sims = {}
    for p in tqdm(wav_paths, desc="speaker sim"):
        ref = ref_dir / p.name
        if ref.exists():
            sims[p.stem] = float(torch.nn.functional.cosine_similarity(
                embed(p), embed(ref), dim=0).item())
    return sims


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--sentences", type=Path, default=Path("data/eval/eval_sentences.csv"))
    ap.add_argument("--judges", nargs="+", default=[WHISPER_ZERO_SHOT, WHISPER_SWAHILI])
    ap.add_argument("--ref-wav-dir", type=Path, default=None)
    ap.add_argument("--skip-utmos", action="store_true")
    ap.add_argument("--out", type=Path, required=True, help="output path stem (writes .json and .md)")
    args = ap.parse_args()

    with open(args.sentences, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (args.wav_dir / f"{r['id']}.wav").exists()]
    if not rows:
        raise SystemExit(f"No wavs in {args.wav_dir} match ids in {args.sentences}")
    print(f"Scoring {len(rows)} clips from {args.wav_dir}")

    clips = {r["id"]: {"category": r.get("category", "all"),
                       "ref": normalize_for_asr(r["text"])} for r in rows}
    wav_paths = [args.wav_dir / f"{cid}.wav" for cid in clips]

    for judge_id in args.judges:
        judge = load_judge(judge_id)
        for cid in tqdm(clips, desc=f"ASR {judge_id}"):
            hyp = normalize_for_asr(transcribe(judge, args.wav_dir / f"{cid}.wav"))
            clips[cid][judge_id] = {
                "hyp": hyp,
                "wer": round(jiwer.wer(clips[cid]["ref"], hyp), 4),
                "cer": round(jiwer.cer(clips[cid]["ref"], hyp), 4),
            }
        del judge
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not args.skip_utmos:
        for cid, score in utmos_scores(wav_paths).items():
            clips[cid]["utmos"] = round(score, 3)
    if args.ref_wav_dir:
        for cid, sim in speaker_similarity(wav_paths, args.ref_wav_dir).items():
            clips[cid]["speaker_sim"] = round(sim, 4)

    # Aggregate overall and per category.
    def aggregate(subset):
        agg = {}
        for judge_id in args.judges:
            agg[judge_id] = {
                "wer": round(sum(c[judge_id]["wer"] for c in subset) / len(subset), 4),
                "cer": round(sum(c[judge_id]["cer"] for c in subset) / len(subset), 4),
            }
        for key in ("utmos", "speaker_sim"):
            vals = [c[key] for c in subset if key in c]
            if vals:
                agg[key] = round(sum(vals) / len(vals), 3)
        return agg

    by_category = defaultdict(list)
    for c in clips.values():
        by_category[c["category"]].append(c)
    summary = {"overall": aggregate(list(clips.values())),
               "by_category": {cat: aggregate(cs) for cat, cs in sorted(by_category.items())},
               "n_clips": len(clips)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "clips": clips}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    lines = [f"# Eval: {args.wav_dir} ({len(clips)} clips)", "",
             "| Scope | " + " | ".join(f"{j} WER / CER" for j in args.judges) + " | UTMOS | SpkSim |",
             "|---|" + "---|" * (len(args.judges) + 2)]
    for scope, agg in [("overall", summary["overall"])] + list(summary["by_category"].items()):
        cells = [f"{agg[j]['wer']:.3f} / {agg[j]['cer']:.3f}" for j in args.judges]
        cells.append(f"{agg.get('utmos', '—')}")
        cells.append(f"{agg.get('speaker_sim', '—')}")
        lines.append(f"| {scope} | " + " | ".join(cells) + " |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
