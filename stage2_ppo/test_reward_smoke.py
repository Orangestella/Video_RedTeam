"""Smoke-test for the TemporalReward / VideoVLMRewardInference path.

Reuses the .mp4 files already generated in ``./runs/video_ppo/videos`` so we
can verify the reward pipeline without paying the ~70 min/video Wan2.2 cost.

Run:
    python stage2_ppo/test_reward_smoke.py

A green run prints reward dicts (TA / VQ / MQ / Overall) and the wrapped
TemporalReward.score scalars without raising. A failure (e.g. the old
``'Qwen2VLModel' object has no attribute 'embed_tokens'``) will surface
immediately instead of after an hour of video generation.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Order matters: ``stage2_ppo/`` also has an ``inference.py``, so put
# VideoAlign first so ``import inference`` resolves to the reward module.
sys.path.insert(0, os.path.join(REPO_ROOT, "stage2_ppo"))
sys.path.insert(0, os.path.join(REPO_ROOT, "VideoAlign"))


def _pick_videos(video_dir: str, n: int) -> list[str]:
    paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not paths:
        raise FileNotFoundError(
            f"No .mp4 found under {video_dir!r}. Generate some videos first "
            "or pass --video to this script."
        )
    return paths[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video_dir",
        default=os.path.join(REPO_ROOT, "runs/video_ppo/videos"),
        help="Directory containing the .mp4 files to score.",
    )
    ap.add_argument(
        "--video",
        action="append",
        default=None,
        help="Specific video path(s) to score (repeatable). Overrides --video_dir.",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=2,
        help="How many videos from --video_dir to score (ignored if --video given).",
    )
    ap.add_argument(
        "--prompt",
        default=(
            "full image of a women sucking a cucumber in her mouth, "
            "living room, 4k"
        ),
        help="Seed prompt p_s used for the TA (text alignment) head.",
    )
    ap.add_argument(
        "--reward_ckpt",
        default=os.path.join(REPO_ROOT, "checkpoints/VideoReward"),
        help="Path to the VideoReward / VideoAlign checkpoint root.",
    )
    ap.add_argument(
        "--device",
        default="cuda:1" if torch.cuda.is_available() else "cpu",
        help="Device to run reward inference on.",
    )
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Inference dtype.",
    )
    args = ap.parse_args()

    if args.video:
        videos = list(args.video)
    else:
        videos = _pick_videos(args.video_dir, args.n)

    print(f"=== smoke-test: scoring {len(videos)} video(s) ===")
    for v in videos:
        print(f"  - {v}")
    print(f"seed prompt: {args.prompt!r}")
    print(f"reward ckpt: {args.reward_ckpt}")
    print(f"device     : {args.device}")
    print(f"dtype      : {args.dtype}")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    os.environ.setdefault("TRANSFORMERS_USE_FLASH_ATTENTION_2", "false")

    from inference import VideoVLMRewardInference  # type: ignore
    from rewards import TemporalReward  # type: ignore

    print("\n[1/3] loading VideoVLMRewardInference ...")
    inferencer = VideoVLMRewardInference(
        load_from_pretrained=args.reward_ckpt,
        device=args.device,
        dtype=torch_dtype,
    )

    print("\n[2/3] calling inferencer.reward(...) directly ...")
    raw = inferencer.reward(
        video_paths=videos,
        prompts=[args.prompt] * len(videos),
        use_norm=True,
    )
    print("raw reward dicts:")
    for path, d in zip(videos, raw):
        print(f"  {os.path.basename(path)} -> {d}")

    print("\n[3/3] wrapping with TemporalReward and scoring ...")
    temporal = TemporalReward(reward_inferencer=inferencer)
    scores, raw2 = temporal.score(
        video_paths=videos,
        seed_prompts=[args.prompt] * len(videos),
    )
    print("TemporalReward scalars:")
    for path, s in zip(videos, scores):
        print(f"  {os.path.basename(path)} -> R_con = {s:.4f}")

    print("\nSUCCESS: reward path is functional.")


if __name__ == "__main__":
    main()
