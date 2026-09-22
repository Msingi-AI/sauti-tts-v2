"""Push the text front-end tools, the model card and spelling demo clips to the
VoxCPM V2 HF repo in ONE commit. Weights are untouched.

Run on the Leonardo login node (authenticated with `hf auth login`).

    python scripts/push_voxcpm_tools_to_hf.py --repo msingiai/sauti-tts-swahili-voxcpm-v2 [--dry-run]
"""
import argparse
import sys
from pathlib import Path

WORK = Path("/leonardo_work/AIH4A_voiceai/sauti_voxcpm")
CARD = WORK / "hf_voxcpm_model_card.md"
TOOLS = ["desplice.py", "swahili_text.py", "english_text.py", "verbalizer.py", "itn.py"]
CLIPS = WORK / "output/spelling_eval/v2_step_0018000_letters2"   # V2 step_0018000, A/Q letter names
SAMPLES = {
    "samples/spelling_phone_sw.wav": CLIPS / "sp_sw_001.wav",   # Piga simu 0712 345 678 ...
    "samples/spelling_phone_en.wav": CLIPS / "sp_en_001.wav",   # Please call 0722 555 019 ...
    "samples/spelling_code_en.wav": CLIPS / "sp_en_003.wav",    # Your booking reference is KQ-4B7X2.
    "samples/spelling_ticket_sw.wav": CLIPS / "sp_sw_002.wav",  # Nambari ya tiketi yako ni 48213.
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi, whoami

    who = whoami()
    print(f"authenticated as: {who.get('name')} (orgs: {[o['name'] for o in who.get('orgs', [])]})")
    api = HfApi()
    info = api.repo_info(repo_id=args.repo)
    if not info.private:
        sys.exit(f"{args.repo} is PUBLIC; refusing to push without an explicit decision.")
    print(f"repo {args.repo} private={info.private} head={info.sha[:8]}")

    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(CARD))]
    for t in TOOLS:
        src = WORK / "scripts" / t
        if not src.exists():
            sys.exit(f"missing tool: {src}")
        ops.append(CommitOperationAdd(path_in_repo=f"tools/{t}", path_or_fileobj=str(src)))
    for dst, src in SAMPLES.items():
        if not src.exists():
            sys.exit(f"missing sample: {src}")
        ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)))

    for op in ops:
        print(f"  {op.path_in_repo:36} {Path(op.path_or_fileobj).stat().st_size / 1e3:8.1f} kB")
    if args.dry_run:
        print("dry run: nothing pushed")
        return
    api.create_tag(repo_id=args.repo, tag="v2.0", revision=info.sha, exist_ok=True)
    print(f"tagged v2.0 -> {info.sha[:8]} (state before this update)")
    commit = api.create_commit(
        repo_id=args.repo, operations=ops,
        commit_message="v2.1: text front-end (verbalizer, ITN, English tables), "
                       "spelling eval in card, spelling demo clips",
    )
    print(f"pushed: {commit.commit_url}")
    api.create_tag(repo_id=args.repo, tag="v2.1", exist_ok=True)
    print("tagged v2.1 -> main")


if __name__ == "__main__":
    main()
