---
language:
- sw
- en
tags:
- text-to-speech
- tts
- swahili
- english
- code-switching
- voxcpm
- voice-cloning
library_name: voxcpm
pipeline_tag: text-to-speech
---

# Sauti TTS V2 — Bilingual Swahili + English (VoxCPM2 full SFT)

**Version 2** of Msingi-AI's Swahili text-to-speech model — now bilingual.
Full SFT of VoxCPM2 on a 723-hour mixture: ~480 h curated Swahili, 180 h
studio English (LibriTTS-R), and an upweighted genuine code-switch subset.
Zero-shot voice cloning from ~5 s of reference audio.

**What V2 fixes over V1:** V1 (Swahili-only training) catastrophically forgot
the base model's English — English words came out mangled. V2 trains fresh from
base with English in the mixture from step one. English is restored to near
perfect while Swahili *improved*.

> **Internal / private.** Licensing audit in `RELEASE_LICENSING.md`
> (Apache-2.0 base, CC-BY-4.0/MIT data): proprietary use is permitted;
> attribution obligations apply to public-facing docs.

---

## ⚠️ Rule 1: clone with prompt + reference — BOTH, always

```python
wav = model.generate(
    text=text,
    prompt_wav_path=REF_WAV,      # the voice to clone
    prompt_text=REF_TRANSCRIPT,   # its exact transcript
    reference_wav_path=REF_WAV,   # same file again
    ...
)
```

**Do NOT use `reference_wav_path` alone.** On these fine-tuned checkpoints it
does not clone: output collapses to a random generic voice per generation
(measured: 92 Hz male reference → 164 Hz output; identity re-rolls every call).
Prompt-mode holds the speaker exactly (F0 within a few Hz, 100% match across
264 eval clips). An earlier version of this card recommended reference-only —
**that guidance was wrong and is reversed.**

ASR metrics **cannot** catch this failure (they don't care whose voice speaks).
Any pipeline change must be validated with an F0/speaker-similarity check, not
WER/CER alone — `scripts/eval_v2.py` in the Sauti repo does this.

## ⚠️ Rule 2: bridge the prompt-splice clicks with `desplice`

Prompt-mode generation leaves 1–2 isolated click transients right after the
last word (the splice where the prompt segment was cut). Bridge them with
`tools/desplice.py` (shipped in this repo — ~1 ms interpolation over isolated
spikes; plosives untouched):

```python
from desplice import desplice
wav, n_fixed = desplice(wav, sr)
```

Do not "fix" clicks with fades or trimming — the transient is inside the
waveform, not at the file edge. (We tried. At length.)

## ⚠️ Rule 3: never send raw digits or bare acronyms — run `verbalize` first

Feed the model **words, not digits**. `2027` as digits produces English-style
year reading ("ishirini, ishirini na saba"); verbalized it is spoken correctly
("elfu mbili ishirini na saba"). `tools/verbalizer.py::verbalize` decides
*how* each span should be read, in the language of the words around it:

| Input | Spoken as |
|---|---|
| `Piga simu 0712 345 678` | sifuri saba moja mbili, tatu nne tano, sita saba nane |
| `Your ticket number is 48213` | four eight two one three |
| `Tiketi yako ni KQ-4B7X2` | kei kyuu, nne bii saba eksi mbili |
| `The KRA office` | the kay ar ey office |
| `shilingi 3,500 kwa kilo` | shilingi elfu tatu mia tano kwa kilo |
| `In 1964` / `Mwaka 1964` | nineteen sixty four / elfu moja mia tisa sitini na nne |

Phone numbers, PINs, account and ticket numbers and alphanumeric codes are read
digit by digit with group pauses; quantities, money, percentages, times and
dates stay cardinals; 2–4 letter capitals are spelled unless they are
pronounced words (NEMA, KEMRI, SACCO ...). Markup overrides the heuristics:
`[spell]…[/spell]`, `[num]…[/num]`, `[say]…[/say]`, `[sw]…[/sw]`, `[en]…[/en]`.
`normalize_for_tts` in `tools/swahili_text.py` now routes through the same code.

`tools/itn.py::inverse_normalize` is the inverse (spoken "sifuri saba moja" →
`071`, "kay queue" → `KQ`) for scoring, or for an ASR front-end that must
write figures rather than number words.

## Recommended usage (the evaluated configuration)

```python
from voxcpm import VoxCPM
from desplice import desplice          # tools/ in this repo
from verbalizer import verbalize       # tools/ in this repo (needs swahili_text.py + english_text.py beside it)

model = VoxCPM.from_pretrained("msingiai/sauti-tts-swahili-voxcpm-v2",
                               load_denoiser=False)

wav = model.generate(
    text=verbalize(text),                # language per span; verbalize(text, lang="sw"|"en") to force
    prompt_wav_path="voice.wav",         # ~5-9 s clean speech of the voice
    prompt_text="<exact transcript of voice.wav>",
    reference_wav_path="voice.wav",
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=True,
    denoise=False,
    retry_badcase=True,
    retry_badcase_max_times=2,
)
wav, _ = desplice(wav, 48000)
# -> float32 numpy, 48 kHz mono
```

**The reference clip controls more than timbre — it also sets the speaking
pace.** A slow, deliberate reference produces slow speech. Use a briskly-read,
clean, 5–9 s clip whose transcript you know exactly.

---

## Evaluation (V2, checkpoint step_0018000)

66-sentence tri-lingual set; category-appropriate judges (Swahili-tuned
Jacaranda ASR for Swahili; whisper-large-v3 `en` for English; whisper-large-v3
auto-detect for code-switch — forcing Swahili falsely penalizes correct
English). F0-match = output pitch within 20% of the reference speaker.

| Category | n | WER | CER | F0-match |
|---|---|---|---|---|
| **Plain Swahili** | 36 | **0.040** | **0.006** | 100% |
| **English** | 12 | **0.022** | **0.012** | 100% |
| Code-switch | 12 | 0.226 | 0.049 | 100% |
| Short words | 6 | 0.167 | 0.033 | 100% |

Versus V1 (Swahili-only): plain-SW CER improved 0.009 → 0.006; English went
from unusable to 2.2% WER; voice fidelity went from unmeasured to a release
gate at 100%.

### Spelling: phone numbers, codes, acronyms (2026-09-22)

30 extra sentences (12 Swahili, 9 English, 9 code-switched) with phone
numbers, ticket and booking codes, PINs, plates and acronyms, verbalized by
`tools/verbalizer.py`. Scored by recall of each item after inverse text
normalization of the whisper-large-v3 transcript: the number, code or acronym
must come back exactly.

| Model | Item accuracy | Swahili | English | Code-switch |
|---|---|---|---|---|
| **V2 step_0018000 (this repo)** | **83%** (34/41) | 82% | 75% | 92% |
| V1 step_0020000 | 68% | 47% | 58% | 67% |

Every phone number, account number and digit-string identifier was read back
exactly on V2. The misses are letter names: English acronyms read letter by
letter (KRA, NSSF) and fast Swahili letter+digit codes. Verbalized text with
words the model already knows solves digits without retraining; isolated
English letter names are the open item.

## Training

| | |
|---|---|
| Base | openbmb/VoxCPM2 (Apache-2.0), trained **fresh from base** |
| Steps | 30,000 (~4.9 epochs), cosine to 0, 4×A100-64GB, 17 h |
| Mixture | 95,669 SW rows (480.5 h) + 88,325 EN rows (180 h, LibriTTS-R) + code-switch rows ×3 |
| Shipped checkpoint | **step_0018000** — best plain-SW + short-word on the sweep; 11k/20k/30k close behind |
| Data fix | ~7% of Swahili transcripts had English spans wrapped in `cs...cs` marker junk (e.g. `csjeanscs`), teaching the model to mangle English; markers stripped (8,466 spans recovered as genuine code-switch data), unfixable rows excluded |

## Limitations

- **Pace follows the reference.** Slow reference → slow speech. Use
  briskly-paced reference clips. (Pace-control experiments ongoing.)
- **Code-switch** works (CER 0.049) but remains the weakest category.
- **Spelled English letter names** (KRA, NSSF read letter by letter) are
  unreliable; digits and Swahili-spelled codes are fine. Prefer `[say]` for
  acronyms that can be pronounced as words.
- **Long-form** (multi-paragraph) is stitched per sentence today, not validated
  end-to-end.
- **No MOS panel yet**; quality confirmed by native-listener review.
- Warmth/preference tuning not yet applied — planned as a post-SFT pass.

## Files

| File | Purpose |
|---|---|
| `model.safetensors`, `audiovae.pth`, `config.json`, tokenizer files | inference weights (step_0018000) |
| `tools/desplice.py` | splice-click bridging (Rule 2) |
| `tools/verbalizer.py` | text front-end: digits, phone numbers, codes, acronyms, money, dates → words, per-span SW/EN (Rule 3) |
| `tools/swahili_text.py`, `tools/english_text.py` | number and letter-name tables used by the verbalizer |
| `tools/itn.py` | inverse text normalization for ASR output (words → figures and codes) |
| `samples/` | demo clips: plain + code-switch × 3 cloned voices; `samples/spelling_*` phone number, ticket code |

## Version history

- **v2.1 (2026-09)** — same weights; new text front-end (`tools/verbalizer.py`,
  `tools/itn.py`, `tools/english_text.py`) for phone numbers, codes and
  acronyms in either language; spelling eval added (83% item accuracy).
- **v2 (2026-08)** — bilingual SW+EN mixture, fresh-from-base; cs-marker data
  bug fixed; prompt-mode + desplice as the blessed path; tri-lingual eval with
  F0 fidelity gate. **Current.**
- v1 (2026-07) — Swahili-only SFT (20k steps). Leaderboard-topping Swahili
  (CER 0.009) but English broken and cloning guidance wrong.

## Contact

Msingi-AI — Sauti TTS project.
