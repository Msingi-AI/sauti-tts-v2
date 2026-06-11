# Sauti TTS V2 — Base Model Selection Research

**Date:** 2026-06-11
**Status:** Research report — decision input for V2 architecture
**Question:** What should Sauti V2 be built on? Specifically: is fine-tuning
[nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) on the WAXAL Swahili
dataset a sound plan?

---

## TL;DR

**Dia-1.6B is not the right base for V2.** It is English-only, frozen (superseded by
Dia2, also English-only), has no official fine-tuning code, no LoRA/Unsloth support,
and zero documented successful low-data non-English fine-tunes — the one quantified
community attempt (~3,000 hours) produced broken output.

**Recommended direction instead (ranked):**

1. **Chatterbox Multilingual (Resemble AI)** — MIT code *and* weights, ~500M params,
   **Swahili already one of its 23 supported languages**. V2 becomes a *quality*
   fine-tune on WAXAL studio data, not teaching a new language from scratch.
2. **Spark-TTS-0.5B** (warm-start from `Sunbird/spark-tts-salt`) — Apache-2.0,
   Unsloth-supported, and the only model family with a published East-African
   fine-tune **including Swahili**, trained on only ~5,000 studio sentences/language.
3. **Continue F5-TTS (the V1 path)** — keep the existing codebase, but note the
   pretrained checkpoint is CC-BY-NC (non-commercial taint) and new-language
   fine-tunes historically used 200–800+ hours.

Run 1 and 2 as a head-to-head bake-off on WAXAL `swa_tts`; ship the winner. Either
fits a single 24 GB GPU.

---

## 1. Where V1 actually stands

Verified from the public repos ([Msingi-AI/sauti-tts](https://github.com/Msingi-AI/sauti-tts),
[Msingi-AI/Sauti](https://github.com/Msingi-AI/Sauti)):

- The original `Sauti` repo was a **design doc** planning an XTTS-v2 fine-tune; the
  actual `sauti-tts` (V1) pivoted to **F5-TTS v1 Base** (flow-matching DiT + Vocos
  vocoder, 24 kHz) fine-tuned on Google **WaxalNLP `swa_tts`**.
- V1 is a *training codebase*, not a shipped model: the MODEL_CARD states **no
  pre-trained weights are bundled**, the evaluation section is an empty template,
  and there is no public demo audio.

**Implication:** V2 is effectively the first model MsingiAI will actually ship. The
bar is therefore not "beat V1" — it is "beat the existing public Swahili baselines"
(`facebook/mms-tts-swh`, bookbot's OpenBible VITS, Chatterbox-MTL zero-shot Swahili),
and **publish weights + eval numbers + demo audio**, which no one has done well for
Swahili. There is currently **no published Swahili TTS benchmark at all** — V2 can
own that gap.

## 2. The data situation

### WAXAL (google/WaxalNLP) — primary training set
- Google's African speech corpus (~21 languages, released early 2026). Two parts:
  ~1,250 h transcribed ASR speech, and a **TTS subset of studio-quality,
  phonetically balanced read speech** (~180–235 h total across languages; figures
  differ between the HF card and arXiv v1 — confirm against
  [arXiv:2602.02734](https://arxiv.org/abs/2602.02734)).
- Recorded by 72 contracted voice actors (36F/36M), target ~16 h clean audio per
  actor → expect roughly **2–4 Swahili voices, ~30–60 h** (per-language table in the
  paper needs confirming; HF was unreachable from this sandbox).
- **License: CC-BY-4.0 — commercial use allowed.** This is the rare purpose-recorded
  studio Swahili TTS resource; it is the right primary dataset. ✅ Your instinct to
  reuse it is correct.

### Supplements worth adding
| Dataset | Content | License | Use |
|---|---|---|---|
| [bookbot/OpenBible_Swahili](https://huggingface.co/datasets/bookbot/OpenBible_Swahili) | Single-narrator studio Bible audio, verse-aligned | CC-BY-SA 4.0 | Extra clean single-speaker hours |
| Makerere/Lacuna curated CV subset ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12337013/)) | 15.1 h, 6 curated female Common Voice speakers, DEMUCS + WV-MOS filtered | open | Speaker diversity; proven for Kiswahili VITS |
| Common Voice Kiswahili | ~400+ validated h, crowdsourced | CC0 | Mine with Whisper-filtering for Tier-2 scale |
| WAXAL `swa_asr` | ~natural transcribed speech | CC-BY-4.0 | ASR-filtered supplement if more hours needed |

Swahili-specific advantages: highly phonemic Latin orthography (character-level
input works; espeak-ng and gruut both have Swahili G2P), non-tonal. Main text-side
work: a small **normalization layer** for numbers/dates (noun-class agreement:
"Mlango 1" → "Mlango wa kwanza") — no mature off-the-shelf tool exists.

## 3. Why not Dia-1.6B (direct answer to the proposal)

What's attractive: Apache-2.0, expressive dialogue (`[S1]`/`[S2]`, `(laughs)`),
byte-level text vocab (256) so Swahili text needs no tokenizer surgery.

What kills it for our use case (all primary-sourced):

1. **English-only pretraining, no multilingual roadmap.** Stated on the model card;
   the Nov-2025 successor Dia2 is *also* English-only, and Dia-1.6B itself is frozen
   (last code commit July 2025).
2. **No training story.** No official fine-tuning code (issues #13/#48/#117 closed
   without release), no LoRA recipe anywhere, Unsloth support still pending. The only
   community pipeline ([stlohrey/dia-finetuning](https://github.com/stlohrey/dia-finetuning))
   is full-fine-tune-only — a 48 GB-GPU regime, and exactly where a small dataset
   causes catastrophic forgetting.
3. **No low-resource precedent.** The single released non-English fine-tune
   (Vietnamese) used undisclosed data on an RTX A6000 with no quality metrics. A
   documented attempt with **~3,000 hours** produced near-zero-length/noisy outputs
   ([issue #27](https://github.com/stlohrey/dia-finetuning/issues/27), unanswered).
   We have perhaps 30–60 h of Swahili — two orders of magnitude less.
4. **Production liabilities even in English:** unstable speaker identity across runs
   (issue #109), ~30 s generation cap.

**Verdict:** the byte tokenizer makes Dia *possible* for Swahili in principle, but
every empirical signal says it needs hundreds–thousands of hours plus big GPUs, with
no successful precedent. With WAXAL-scale data, it's the wrong bet.

## 4. Candidate comparison

| Base | License (weights) | Params | Swahili status | Fine-tune story | Verdict |
|---|---|---|---|---|---|
| **Chatterbox Multilingual** | **MIT** ✅ (verified LICENSE) | ~500M | **Supported out of the box** ✅ (verified README) | Community LoRA + vocab-extension toolkits; Slovak low-resource precedent; no official scripts | **Top pick** |
| **Spark-TTS-0.5B** | Apache-2.0 | 0.5B | Via `Sunbird/spark-tts-salt` (7 EA languages incl. Swahili, ~5k studio sentences each) | Unsloth-supported; official training code still unreleased | **Strong #2 — proven EA precedent** |
| **F5-TTS** (V1 path) | Code MIT, **checkpoint CC-BY-NC** ⚠️ | 336M | No Swahili fine-tune exists | Best community new-language registry, but typical recipes used 200–800 h; <5 h only works with ASR-reward RL (Align2Speak) | Keep as fallback; license taint |
| Orpheus-3B | Apache-2.0 | 3B | None; 7-language preview, no African langs | Best LoRA/Unsloth ecosystem; maintenance slowed mid-2025 | Viable but heaviest |
| CosyVoice 2/3 | Apache-2.0 | 0.5B | None | Best *official* training stack; ~10 h → MOS ≈ 2.9 curve | Runner-up |
| CSM-1B (Sesame) | Apache-2.0 | 1B | None; English-centric by design | Speechmatics new-language guide; Unsloth support | Possible, weaker precedent |
| **Dia-1.6B** | Apache-2.0 | 1.6B | None; English-only | None official; failure reports | **Rejected (§3)** |
| XTTS-v2 | **CPML non-commercial, unobtainable** ❌ | 467M | Not in 17 languages | Community recipes ~100+ h | Rejected (license) |
| MMS-TTS-swh | CC-BY-NC ⚠️ | small VITS | Ships Swahili | HF VITS fine-tuning works | **Use as the baseline to beat**, not the base |
| Fish/OpenAudio, OuteTTS-1B, IndexTTS-2, Higgs | NC / custom ❌ | — | — | — | Rejected (license/no training code) |
| VITS from scratch | MIT (your weights) | ~40M | Makerere built Kiswahili from 15.1 h | From-scratch on 24 GB GPU, days | Cheap insurance baseline |

## 5. Proposed V2 plan

**Phase 0 — Data + eval harness first (1–2 weeks, CPU-mostly).**
- Confirm WAXAL `swa_tts` per-speaker hours; build manifest, 24 kHz resample,
  Whisper-large-v3 transcript-agreement filter (drop CER > ~10%).
- Hold out a fixed eval set (~200 sentences incl. numbers, loanwords, code-switching).
- Build the eval harness *before* training (see §6). Score the public baselines
  (`facebook/mms-tts-swh`, bookbot VITS, Chatterbox-MTL zero-shot) so every later
  checkpoint has context.

**Phase 1 — Bake-off (2–4 weeks, single 24 GB GPU, ~$100–300).**
- Track A: Chatterbox-MTL fine-tune on WAXAL (community toolkit, LoRA first, then
  full decoder if LoRA under-adapts).
- Track B: Spark-TTS warm-started from `Sunbird/spark-tts-salt` (or base) via Unsloth.
- Same data, same eval. Note for codec-LLM language adaptation, community consensus
  is **full fine-tune > LoRA for language shifts**; LoRA is fine here since Swahili
  is already in-distribution for both tracks.

**Phase 2 — Ship the winner.**
- Scale data (OpenBible + Makerere subset + filtered Common Voice) if eval says
  data-bound; add Swahili number/date normalizer; publish weights, eval table, and
  demo audio on HF under `msingiai/` — that alone exceeds every prior Swahili TTS
  release.

## 6. Evaluation protocol (the V1 gap to fix)

1. **Intelligibility:** ASR-WER **and CER** (Swahili's agglutinative morphology
   inflates WER) with **two judges**: zero-shot Whisper-large-v3 + a Swahili-tuned
   Whisper (e.g. Jacaranda-Health/ASR-STT or microsoft/paza-whisper-large-v3-turbo).
   Judge floor ≈ its own ~10% WER on real speech.
2. **Speaker similarity:** ECAPA-TDNN / WavLM cosine vs reference.
3. **Quality proxy:** UTMOS v1 as *relative* signal only (English-trained; not
   calibrated for Swahili). NISQA-TTS as second opinion.
4. **Human MOS:** ~100 held-out sentences, ≥10 native listeners (Makerere precedent),
   vs `mms-tts-swh` and Chatterbox zero-shot as anchors.
5. Stress sets: numbers/dates, English code-switching, named entities.

## 7. Open items to verify (HF unreachable from research sandbox)

- [ ] WAXAL `swa_tts` exact hours/speakers (per-language table in arXiv:2602.02734).
- [ ] `Sunbird/spark-tts-salt` checkpoint quality — listen before warm-starting.
- [ ] Chatterbox Swahili zero-shot quality on WAXAL voices (no per-language MOS published).
- [ ] OpenBible_Swahili total hours.

## Key sources

- Dia: [github.com/nari-labs/dia](https://github.com/nari-labs/dia) (README, config.py),
  [stlohrey/dia-finetuning](https://github.com/stlohrey/dia-finetuning) (+ issues #22/#23/#27),
  [unsloth#2410](https://github.com/unslothai/unsloth/issues/2410)
- Chatterbox: [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
  (README + LICENSE fetched directly), [gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning)
- Spark-TTS/Sunbird: [github.com/SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS),
  [github.com/SunbirdAI/salt](https://github.com/SunbirdAI/salt) (fetched: "~5,000 sentences
  read out by professional voice actors", Swahili listed),
  [Sunbird/spark-tts-salt](https://huggingface.co/Sunbird/spark-tts-salt)
- V1/WAXAL: [Msingi-AI/sauti-tts](https://github.com/Msingi-AI/sauti-tts),
  [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP),
  [arXiv:2602.02734](https://arxiv.org/abs/2602.02734)
- Swahili TTS prior art: [facebook/mms-tts-swh](https://huggingface.co/facebook/mms-tts-swh),
  [bookbot/vits-base-sw-KE-OpenBible](https://huggingface.co/bookbot/vits-base-sw-KE-OpenBible),
  Katumba et al. 2025 ([Wiley](https://onlinelibrary.wiley.com/doi/10.1002/ail2.117),
  [PMC12337013](https://pmc.ncbi.nlm.nih.gov/articles/PMC12337013/))
- Methodology: [Align2Speak, arXiv:2509.21718](https://arxiv.org/pdf/2509.21718),
  [F5-TTS SHARED.md](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/SHARED.md),
  [anhnh2002/XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages),
  [TTSDS2](https://arxiv.org/pdf/2506.19441)
