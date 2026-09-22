#!/bin/bash
#SBATCH --job-name=spelling_eval
#SBATCH --account=aih4a_voiceai
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/leonardo_work/AIH4A_voiceai/sauti_voxcpm/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIH4A_voiceai/sauti_voxcpm/logs/%x-%j.err

# Spelling eval (phone numbers, ticket codes, acronyms, PINs): synth the 30
# `spelling` rows of eval_sentences_v2.csv through the new verbalizer, in
# prompt-mode cloning + desplice, on V1 step_0020000 (Swahili-only) and V2
# step_0018000 (shipped SW+EN model); score with ITN item accuracy.

set -euo pipefail
WORK=/leonardo_work/AIH4A_voiceai/sauti_voxcpm
cd "$WORK"
module load python/3.11.7 cuda/12.2 nccl
source venvs/voxcpm/bin/activate
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

PROMPT_WAV="$WORK/data/sw_voxcpm_sources_v1/waxal_swa_tts_chunk_0/audio/waxal_swa_tts/train/swa_817_917e3d1c8e7d.flac"
PROMPT_TEXT="Jua leo halikuonekana kwa sababu mvua ilinyesha siku nzima."
SENT="$WORK/data/eval/eval_sentences_v2.csv"

python - <<'PY'
import sys; sys.path.insert(0, "scripts")
from verbalizer import verbalize
import csv
rows=[r for r in csv.DictReader(open("data/eval/eval_sentences_v2.csv",encoding="utf-8")) if r["category"]=="spelling"]
print(f"{len(rows)} spelling rows; verbalized forms:")
for r in rows: print(" ", r["id"], "|", verbalize(r["text"]))
PY

ids_by_prefix() { python -c "
import csv,sys
print(','.join(r['id'] for r in csv.DictReader(open('$SENT',encoding='utf-8')) if r['id'].startswith(sys.argv[1])))" "$1"; }
SW_IDS=$(ids_by_prefix sp_sw); EN_IDS=$(ids_by_prefix sp_en); CS_IDS=$(ids_by_prefix sp_cs)

run_ckpt() {
  local TAG="$1" CKPT="$2"
  local OUT="$WORK/output/spelling_eval/$TAG"
  echo "=== $TAG: $CKPT ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  python scripts/synth_voxcpm.py --sentences "$SENT" --out "$OUT" --model-dir "$CKPT" \
    --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT" --ids "$SW_IDS" --max-attempts 4 --check-language sw
  python scripts/synth_voxcpm.py --sentences "$SENT" --out "$OUT" --model-dir "$CKPT" \
    --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT" --ids "$EN_IDS" --max-attempts 4 --check-language en
  python scripts/synth_voxcpm.py --sentences "$SENT" --out "$OUT" --model-dir "$CKPT" \
    --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT" --ids "$CS_IDS" --max-attempts 4 --check-language auto
  echo "=== $TAG: score ==="
  python scripts/eval_v2.py --wav-dir "$OUT" --sentences "$SENT" --ref-wav "$PROMPT_WAV" \
    --out "$WORK/benchmarks/results/spelling_$TAG"
}

run_ckpt v2_step_0018000 "$WORK/runs/voxcpm2_swen_sft_v2/checkpoints/step_0018000"
run_ckpt v1_step_0020000 "$WORK/runs/voxcpm2_sw_full_sft_v1/checkpoints/step_0020000"

echo "=== SPELLING EVAL DONE ==="
for TAG in v2_step_0018000 v1_step_0020000; do echo "--- $TAG ---"; cat "$WORK/benchmarks/results/spelling_$TAG.md"; done
