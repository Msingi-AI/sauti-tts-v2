"""Tri-lingual eval: score synthesized wavs per category with the right judge.

Category -> judge/language:
  english      -> whisper-large-v3, language='en'
  code_switch  -> whisper-large-v3, language=None (auto; forcing 'sw' falsely
                  penalizes correctly-spoken English -- prior Dia finding)
  spelling     -> whisper-large-v3, 'en' for sp_en_* ids, auto otherwise
                  (spelled letters and digit strings are mostly English-shaped)
  everything else -> Jacaranda Swahili ASR, language='sw'

The spelling category (phone numbers, ticket codes, acronyms, PINs) is scored
by running the hypothesis through inverse text normalization (itn.py) and
checking that every item the sentence contains (0712345678, KQ4B7X2, KRA)
comes back: item_acc (recall) and item_cer (char error of the closest match).
Plain WER/CER on these clips is computed on the ITN'd hypothesis so a
correctly spelled number is not penalized for Whisper writing it as figures.

Also measures F0 voice fidelity vs the reference (v1 lesson: intelligibility
metrics cannot detect cloning failure).

  python eval_v2.py --wav-dir <dir> --sentences <csv> --ref-wav <wav> --out <stem>
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np

import sys, os
sys.path.insert(0, str(Path(__file__).parent))
from asr_judge import WHISPER_SWAHILI, WHISPER_ZERO_SHOT, load_judge, transcribe
from build_reference_voices import estimate_f0, _resample
from itn import extract_items, inverse_normalize
from swahili_text import normalize_for_asr
from verbalizer import expected_items


def item_hit(expected: str, got: list[str]) -> bool:
    """An expected item counts if it is a got item, is contained in one (KDA in
    KDA123B; 3+ chars only), or is a contiguous concatenation of got items (KQ + 4B7X2)."""
    if expected in got:
        return True
    if len(expected) >= 3 and (any(expected in g for g in got) or expected in "".join(got)):
        return True
    for i in range(len(got)):
        acc = ""
        for j in range(i, len(got)):
            acc += got[j]
            if acc == expected:
                return True
            if len(acc) >= len(expected):
                break
    return False


def item_cer(expected: str, got: list[str]) -> float:
    if not got:
        return 1.0
    candidates = got + ["".join(got)]
    return min(min(jiwer.cer(expected, g), 1.0) for g in candidates)


def score_spelling(text: str, raw_hyp: str) -> dict:
    itn_hyp = inverse_normalize(raw_hyp)
    expected = expected_items(text)
    got = extract_items(itn_hyp, already_normalized=True)
    hits = [item_hit(e, got) for e in expected]
    cers = [0.0 if h else item_cer(e, got) for e, h in zip(expected, hits)]
    return {
        "itn_hyp": itn_hyp, "expected_items": expected, "got_items": got,
        "item_hits": int(sum(hits)), "item_total": len(expected),
        "item_acc": round(sum(hits) / len(expected), 3) if expected else 1.0,
        "item_cer": round(float(np.mean(cers)), 4) if expected else 0.0,
    }


def f0_of(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    xa, sra = _resample(x, sr, 24000)
    f0, _, _ = estimate_f0(xa, sra)
    return f0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--sentences", type=Path, required=True)
    ap.add_argument("--ref-wav", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.sentences, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if (args.wav_dir / f"{r['id']}.wav").exists()]
    print(f"scoring {len(rows)} clips from {args.wav_dir}")

    ref_f0 = f0_of(args.ref_wav)
    print(f"reference F0: {ref_f0:.1f} Hz")

    judges = {}

    def get_judge(mid):
        if mid not in judges:
            judges[mid] = load_judge(mid)
        return judges[mid]

    clips = {}
    for r in rows:
        cat = r.get("category", "general")
        cid = r["id"]
        wav = args.wav_dir / f"{cid}.wav"
        if cat == "english":
            judge_id, lang = WHISPER_ZERO_SHOT, "en"
        elif cat == "code_switch":
            judge_id, lang = WHISPER_ZERO_SHOT, None
        elif cat == "spelling":
            judge_id, lang = WHISPER_ZERO_SHOT, ("en" if cid.startswith("sp_en") else None)
        else:
            judge_id, lang = WHISPER_SWAHILI, "sw"
        raw_hyp = transcribe(get_judge(judge_id), wav, language=lang)
        spelling = score_spelling(r["text"], raw_hyp) if cat == "spelling" else {}
        hyp = normalize_for_asr(spelling["itn_hyp"] if spelling else raw_hyp)
        ref = normalize_for_asr(r["text"])
        f0 = f0_of(wav)
        f0_ok = ref_f0 > 0 and f0 > 0 and abs(f0 - ref_f0) / ref_f0 < 0.20
        clips[cid] = {
            "id": cid, "category": cat, "judge": judge_id, "lang": lang or "auto",
            "ref": ref, "hyp": hyp, "raw_hyp": raw_hyp,
            "wer": round(jiwer.wer(ref, hyp), 4) if ref else 1.0,
            "cer": round(jiwer.cer(ref, hyp), 4) if ref else 1.0,
            "f0": round(f0, 1), "f0_match": bool(f0_ok),
            **spelling,
        }
        extra = f" items={spelling['item_hits']}/{spelling['item_total']}" if spelling else ""
        print(f"  {cid} [{cat}] cer={clips[cid]['cer']:.3f}{extra} f0={f0:.0f} "
              f"({'ok' if f0_ok else 'DRIFT'})", flush=True)

    by_cat = defaultdict(list)
    for c in clips.values():
        by_cat[c["category"]].append(c)

    summary = {}
    for cat, cs in sorted(by_cat.items()):
        summary[cat] = {
            "n": len(cs),
            "wer": round(float(np.mean([c["wer"] for c in cs])), 4),
            "cer": round(float(np.mean([c["cer"] for c in cs])), 4),
            "f0_match_rate": round(float(np.mean([c["f0_match"] for c in cs])), 3),
        }
        if cat == "spelling":
            total = sum(c["item_total"] for c in cs)
            summary[cat]["item_acc"] = round(sum(c["item_hits"] for c in cs) / total, 3) if total else 1.0
            summary[cat]["item_cer"] = round(float(np.mean([c["item_cer"] for c in cs])), 4)
            for sub in ("sp_sw", "sp_en", "sp_cs"):
                sub_cs = [c for c in cs if c["id"].startswith(sub)]
                sub_total = sum(c["item_total"] for c in sub_cs)
                if sub_total:
                    summary[cat][f"item_acc_{sub[3:]}"] = round(
                        sum(c["item_hits"] for c in sub_cs) / sub_total, 3)
    plain = [c for c in clips.values()
             if c["category"] not in ("code_switch", "english", "short_word", "spelling")]
    if plain:
        summary["plain_sw_overall"] = {
            "n": len(plain),
            "wer": round(float(np.mean([c["wer"] for c in plain])), 4),
            "cer": round(float(np.mean([c["cer"] for c in plain])), 4),
            "f0_match_rate": round(float(np.mean([c["f0_match"] for c in plain])), 3),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "ref_f0": ref_f0, "clips": clips},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Eval v2: {args.wav_dir}", "",
             "| category | n | WER | CER | F0-match | item acc | item CER |",
             "|---|---|---|---|---|---|---|"]
    for cat, s in summary.items():
        items = (f"{s['item_acc']:.1%} | {s['item_cer']:.3f}" if "item_acc" in s else "- | -")
        lines.append(f"| {cat} | {s['n']} | {s['wer']:.3f} | {s['cer']:.3f} | "
                     f"{s['f0_match_rate']:.0%} | {items} |")
    if "spelling" in summary:
        sp = summary["spelling"]
        parts = [f"{k[9:]}={sp[k]:.1%}" for k in ("item_acc_sw", "item_acc_en", "item_acc_cs") if k in sp]
        lines.append("")
        lines.append("Spelling item accuracy by language: " + ", ".join(parts))
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
