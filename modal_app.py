"""Modal app for the Sauti V2 pipeline: data prep -> filter -> baselines -> train -> eval.

Everything reads/writes the persistent `sauti-tts-v2-data` Volume:

  /vol/data/waxal_sw/        prepared WAXAL Swahili (wavs/ + metadata.csv)
  /vol/runs/<name>/          synthesized wavs + eval results per run
  /vol/checkpoints/<name>/   training outputs

Usage (from the repo root, after `pip install modal` and `modal setup`):

  modal run modal_app.py --stage prepare
  modal run modal_app.py --stage filter
  modal run modal_app.py --stage baselines
  modal run modal_app.py --stage train          # Track A: Chatterbox LoRA
  modal run modal_app.py --stage eval --run-name baseline_mms

If WaxalNLP requires authentication, create the secret once:
  modal secret create huggingface HF_TOKEN=hf_...
"""

import subprocess
from pathlib import Path

import modal

app = modal.App("sauti-tts-v2")
vol = modal.Volume.from_name("sauti-tts-v2-data", create_if_missing=True)

VOL = "/vol"
DATA = f"{VOL}/data/waxal_sw"
RUNS = f"{VOL}/runs"
CKPT = f"{VOL}/checkpoints"

# TODO: pin to a commit SHA after the first successful training run.
TOOLKIT_REPO = "https://github.com/gokhaneraslan/chatterbox-finetuning"
TOOLKIT_REF = "main"
TOOLKIT_DIR = "/opt/chatterbox-finetuning"

# Training hardware. Multi-GPU is launched via `accelerate` and only speeds
# things up if the toolkit's train loop supports DDP — watch per-GPU
# utilization on the first run; if only GPU 0 is busy, drop the count to 1.
TRAIN_GPU = "A100-40GB"
TRAIN_GPU_COUNT = 4

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.1", "torchaudio>=2.1", "transformers>=4.46", "accelerate",
        "datasets[audio]>=2.20", "soundfile", "jiwer>=3.0", "speechbrain>=1.0", "tqdm",
    )
    .add_local_dir("scripts", remote_path="/root/sauti/scripts")
    .add_local_dir("data/eval", remote_path="/root/sauti/data/eval")
)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .run_commands(
        f"git clone {TOOLKIT_REPO} {TOOLKIT_DIR}",
        f"cd {TOOLKIT_DIR} && git checkout {TOOLKIT_REF} && pip install -r requirements.txt",
    )
    .pip_install("accelerate")
    .add_local_dir("scripts", remote_path="/root/sauti/scripts")
)

baseline_image = base_image.pip_install("chatterbox-tts")

hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])


def run_script(name: str, *args: str):
    subprocess.run(
        ["python", f"/root/sauti/scripts/{name}", *args],
        cwd="/root/sauti", check=True,
    )


@app.function(image=base_image, volumes={VOL: vol}, secrets=[hf_secret], timeout=4 * 3600)
def prepare_data():
    run_script("prepare_waxal.py", "--out", DATA)
    vol.commit()


@app.function(image=base_image, volumes={VOL: vol}, gpu="L4", timeout=8 * 3600)
def filter_data(max_cer: float = 0.10):
    run_script("filter_transcripts.py", "--data", DATA, "--max-cer", str(max_cer))
    vol.commit()


@app.function(image=baseline_image, volumes={VOL: vol}, gpu="L4", timeout=2 * 3600)
def synth_baselines():
    run_script("synth_baseline_mms.py", "--out", f"{RUNS}/baseline_mms")
    run_script("synth_baseline_chatterbox.py", "--out", f"{RUNS}/baseline_chatterbox")
    vol.commit()


@app.function(image=base_image, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600)
def evaluate(run_name: str):
    run_script(
        "run_eval.py",
        "--wav-dir", f"{RUNS}/{run_name}",
        "--out", f"{RUNS}/{run_name}/results",
    )
    vol.commit()
    print((Path(RUNS) / run_name / "results.md").read_text())


def patch_config(config_path: Path, overrides: dict):
    """Rewrite `key = value` / `key: type = value` lines in the toolkit's src/config.py.

    Fails loudly on unknown keys so upstream config drift surfaces immediately
    instead of silently training with defaults.
    """
    import re

    text = config_path.read_text()
    for key, value in overrides.items():
        pattern = rf"(?m)^(\s*{re.escape(key)}\s*(?::[^=]+)?=\s*).*$"
        if not re.search(pattern, text):
            raise KeyError(
                f"{key!r} not found in {config_path} — toolkit config schema changed; "
                f"inspect src/config.py at {TOOLKIT_REPO}@{TOOLKIT_REF}"
            )
        text = re.sub(pattern, rf"\g<1>{value!r}" if isinstance(value, str) else rf"\g<1>{value}", text)
    config_path.write_text(text)


def patch_dataset_path(config_path: Path, dataset_dir: str):
    """Point the toolkit at our prepared dataset, whatever its config calls the key.

    The exact key name was not verifiable at design time; probing candidates and
    failing loudly beats silently training on the toolkit's default dataset.
    """
    candidates = ("dataset_path", "dataset_dir", "data_path", "data_dir", "metadata_path")
    for key in candidates:
        try:
            patch_config(config_path, {key: dataset_dir})
            print(f"Dataset path set via config key {key!r}")
            return
        except KeyError:
            continue
    raise KeyError(
        f"No dataset-path key found among {candidates} in {config_path}; "
        f"read src/config.py at {TOOLKIT_REPO}@{TOOLKIT_REF} and update this function."
    )


@app.function(image=train_image, volumes={VOL: vol},
              gpu=f"{TRAIN_GPU}:{TRAIN_GPU_COUNT}", timeout=24 * 3600)
def train_chatterbox(lora: bool = True, run_name: str = "chatterbox_sw_lora"):
    """Track A: fine-tune Chatterbox on the filtered WAXAL data.

    First-run checklist (config keys below match the toolkit README; verify
    against src/config.py before a long run):
      - confirm how the toolkit selects the MULTILINGUAL checkpoint (Swahili
        needs ChatterboxMultilingualTTS, not the English Turbo model)
      - confirm the dataset-path config key name
      - with TRAIN_GPU_COUNT > 1, confirm all GPUs show utilization (toolkit
        must support DDP for the extra GPUs to contribute)
    """
    import shutil

    config = Path(TOOLKIT_DIR) / "src" / "config.py"
    metadata = Path(DATA) / "metadata_filtered.csv"
    if not metadata.exists():
        raise FileNotFoundError("Run --stage filter first (metadata_filtered.csv missing)")

    patch_config(config, {
        "is_lora": lora,
        "is_turbo": False,   # standard (Llama-based) path — see docstring re: multilingual
        "ljspeech": True,
        "json_format": False,
        "preprocess": True,
    })
    # The toolkit expects LJSpeech layout (metadata.csv next to wavs/); feed it the
    # filtered manifest by staging a view of the prepared data.
    staged = Path("/tmp/dataset")
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "wavs").symlink_to(Path(DATA) / "wavs")
    (staged / "metadata.csv").write_text(metadata.read_text())
    patch_dataset_path(config, str(staged))

    subprocess.run(["python", "setup.py"], cwd=TOOLKIT_DIR, check=True)

    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        train_cmd = ["accelerate", "launch", "--multi_gpu",
                     f"--num_processes={n_gpus}", "train.py"]
    else:
        train_cmd = ["python", "train.py"]
    subprocess.run(train_cmd, cwd=TOOLKIT_DIR, check=True)

    out = Path(CKPT) / run_name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(TOOLKIT_DIR) / "chatterbox_output", out, dirs_exist_ok=True)
    vol.commit()
    print(f"Checkpoint saved to {out}")


@app.local_entrypoint()
def main(stage: str = "prepare", run_name: str = "", lora: bool = True):
    if stage == "prepare":
        prepare_data.remote()
    elif stage == "filter":
        filter_data.remote()
    elif stage == "baselines":
        synth_baselines.remote()
    elif stage == "train":
        train_chatterbox.remote(lora=lora, run_name=run_name or "chatterbox_sw_lora")
    elif stage == "eval":
        if not run_name:
            raise SystemExit("--run-name required for eval (e.g. baseline_mms)")
        evaluate.remote(run_name)
    else:
        raise SystemExit(f"Unknown stage {stage!r}; use prepare|filter|baselines|train|eval")
