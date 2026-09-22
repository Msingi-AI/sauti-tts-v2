#!/bin/bash
#SBATCH --job-name=spelling_letters2
#SBATCH --account=aih4a_voiceai
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/AIH4A_voiceai/sauti_voxcpm/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIH4A_voiceai/sauti_voxcpm/logs/%x-%j.err

# Re-synth the code/acronym spelling rows on V2 step_0018000 after changing the
# English letter names A -> "ey", Q -> "queue". Phone/number rows are copied
# from the first run (output/spelling_eval/v2_step_0018000) so the score covers
# all 30 rows.

set -euo pipefail
WORK=/leonardo_work/AIH4A_voiceai/sauti_voxcpm
cd "$WORK"
module load python/3.11.7 cuda/12.2 nccl
source venvs/voxcpm/bin/activate
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

PROMPT_WAV="$WORK/data/sw_voxcpm_sources_v1/waxal_swa_tts_chunk_0/audio/waxal_swa_tts/train/swa_817_917e3d1c8e7d.flac"
PROMPT_TEXT="Jua leo halikuonekana kwa sababu mvua ilinyesha siku nzima."
SENT="$WORK/data/eval/eval_sentences_v2.csv"
CKPT="$WORK/runs/voxcpm2_swen_sft_v2/checkpoints/step_0018000"
OLD="$WORK/output/spelling_eval/v2_step_0018000"
OUT="$WORK/output/spelling_eval/v2_step_0018000_letters2"

# rows whose text contains a code or acronym (letters read letter by letter)
read -r SW_IDS EN_IDS CS_IDS <<<"$(python - <<'PY'
import csv, sys; sys.path.insert(0, "scripts")
from verbalizer import classify
rows=[r for r in csv.DictReader(open("data/eval/eval_sentences_v2.csv",encoding="utf-8")) if r["category"]=="spelling"]
sel=[r["id"] for r in rows if any(s.cls in ("code","acronym") for s in classify(r["text"]))]
print(" ".join(",".join(i for i in sel if i.startswith(p)) for p in ("sp_sw","sp_en","sp_cs")))
PY
)"
echo "code rows: sw=$SW_IDS en=$EN_IDS cs=$CS_IDS"

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$OLD"/*.wav "$OUT"/
python scripts/synth_voxcpm.py --sentences "$SENT" --out "$OUT" --model-dir "$CKPT" \
  --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT" --ids "$SW_IDS" --max-attempts 4 --check-language sw
python scripts/synth_voxcpm.py --sentences "$SENT" --out "$OUT" --model-dir "$CKPT" \
  --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT" --ids "$EN_IDS" --max-attempts 4 --check-language en
python scripts/synth_voxcpm.py --sentences "$SENT" --out "$OUT" --model-dir "$CKPT" \
  --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT" --ids "$CS_IDS" --max-attempts 4 --check-language auto
python scripts/eval_v2.py --wav-dir "$OUT" --sentences "$SENT" --ref-wav "$PROMPT_WAV" \
  --out "$WORK/benchmarks/results/spelling_v2_step_0018000_letters2"
echo "=== LETTERS2 DONE ==="
cat "$WORK/benchmarks/results/spelling_v2_step_0018000_letters2.md"
