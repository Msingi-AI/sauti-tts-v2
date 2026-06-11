"""Shared ASR-judge helpers used by transcript filtering and TTS evaluation."""

import torch
from transformers import pipeline

WHISPER_ZERO_SHOT = "openai/whisper-large-v3"
WHISPER_SWAHILI = "Jacaranda-Health/ASR-STT"


def load_judge(model_id: str, device: str | None = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device=device,
        chunk_length_s=30,
    )


def transcribe(judge, wav_path) -> str:
    generate_kwargs = {}
    if "whisper" in judge.model.config.model_type:
        generate_kwargs = {"language": "sw", "task": "transcribe"}
    return judge(str(wav_path), generate_kwargs=generate_kwargs)["text"].strip()
