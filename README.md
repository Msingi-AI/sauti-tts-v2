# Sauti TTS V2

Open Swahili text-to-speech from [MsingiAI](https://github.com/Msingi-AI).

**Status:** Phase 0 — data preparation + evaluation harness.
**Design rationale:** see [docs/research/v2-base-model-selection.md](docs/research/v2-base-model-selection.md).

## Plan

V2 is a head-to-head bake-off of two bases fine-tuned on Google's
[WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) `swa_tts` studio data
(CC-BY-4.0):

- **Track A:** Chatterbox Multilingual (MIT, Swahili in-distribution via `language_id="sw"`)
- **Track B:** Spark-TTS-0.5B (Apache-2.0, warm-start from `Sunbird/spark-tts-salt`)

The winner ships with published weights, eval table, and demo audio. Baselines to
beat: `facebook/mms-tts-swh`, bookbot OpenBible VITS, Chatterbox zero-shot.

## Layout

```
data/eval/eval_sentences.csv   Held-out eval set (starter: 48 sentences, 4 stress categories)
scripts/prepare_waxal.py       WAXAL swa_tts -> LJSpeech-style wavs/ + metadata.csv
scripts/filter_transcripts.py  ASR transcript-agreement (CER) filter on prepared data
scripts/synth_baseline_mms.py        Baseline synthesis: facebook/mms-tts-swh
scripts/synth_baseline_chatterbox.py Baseline synthesis: Chatterbox-MTL zero-shot sw
scripts/run_eval.py            Eval harness: dual ASR judges (WER/CER), UTMOS, speaker sim
scripts/swahili_text.py        Swahili number verbalization + text normalization
scripts/asr_judge.py           Shared ASR-judge helpers
docs/research/                 Research reports
```

## Quickstart

```bash
pip install -r requirements.txt

# 0. Prepare WAXAL Swahili TTS data (inspects schema, writes wavs + manifest)
python scripts/prepare_waxal.py --out data/waxal_sw

# 1. Filter rows whose audio disagrees with the transcript (judge CER > 10%)
python scripts/filter_transcripts.py --data data/waxal_sw

# 2. Synthesize the eval set with the public baselines
python scripts/synth_baseline_mms.py --out runs/baseline_mms
python scripts/synth_baseline_chatterbox.py --out runs/baseline_chatterbox

# 3. Score any run (baseline or our checkpoints) the same way
python scripts/run_eval.py --wav-dir runs/baseline_mms --out runs/baseline_mms/results
```

## Evaluation protocol

- **Intelligibility:** WER **and CER** (Swahili's agglutinative morphology inflates
  WER) under two ASR judges: zero-shot `openai/whisper-large-v3` and a
  Swahili-fine-tuned Whisper (default `Jacaranda-Health/ASR-STT`). A judge's own
  error rate on real speech is the floor — don't over-interpret below it.
- **Quality proxy:** UTMOS (English-trained — treat as a *relative* ranking signal
  only for Swahili).
- **Speaker similarity:** ECAPA-TDNN cosine vs. reference audio (for cloning/fine-tune
  fidelity).
- **Human MOS:** ~100 sentences, ≥10 native listeners, anchored against
  `mms-tts-swh` and Chatterbox zero-shot (run before any release).

`data/eval/eval_sentences.csv` is a 48-sentence starter set across four stress
categories (general, numbers/dates, code-switching, named entities). It needs
native-speaker review and expansion to ~200 sentences before results are quotable.

## License

Code: MIT. Datasets and base-model weights carry their own licenses — see the
research report's license column before redistributing anything trained here.
