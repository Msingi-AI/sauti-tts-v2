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

## Running on Modal

The full pipeline runs on the team's [Modal](https://modal.com) workspace —
no local GPU needed. Data, checkpoints, and results persist in the
`sauti-tts-v2-data` Volume.

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_...   # once, if WaxalNLP is gated

modal run modal_app.py --stage prepare      # CPU: WAXAL -> Volume
modal run modal_app.py --stage filter       # L4: CER transcript filter
modal run modal_app.py --stage baselines    # L4: MMS + Chatterbox zero-shot
modal run modal_app.py --stage eval --run-name baseline_mms
modal run modal_app.py --stage train        # A10G: Chatterbox LoRA (Track A)
modal run modal_app.py --stage eval --run-name chatterbox_sw_lora
```

Ballpark cost: prepare + filter + baselines + eval ≈ a few GPU-hours on L4
(~$1/h). Training runs on 4×A100-40GB (~$8/h total; set via `TRAIN_GPU` /
`TRAIN_GPU_COUNT` in `modal_app.py`) and is launched through `accelerate` —
on the first run, confirm all four GPUs show utilization; if the fine-tuning
toolkit turns out not to support DDP, drop `TRAIN_GPU_COUNT` to 1 (~$2/h)
rather than paying for idle GPUs.

Before the first long training run, check the items in
`train_chatterbox()`'s docstring — the fine-tuning toolkit's config schema
was not fully verifiable at design time, and the script is built to fail
loudly (not silently mistrain) if keys have drifted.

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
