#!/usr/bin/env python3
"""Merge the latest SFT LoRA adapter into the Llama-3 base model."""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def checkpoint_number(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def has_adapter(path: Path) -> bool:
    return (
        (path / "adapter_config.json").is_file()
        and ((path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file())
    )


def find_latest_adapter(sft_dir: Path) -> Path:
    candidates = [p for p in sft_dir.glob("checkpoint-*") if p.is_dir() and has_adapter(p)]
    if candidates:
        return max(candidates, key=checkpoint_number)
    if has_adapter(sft_dir):
        return sft_dir
    raise FileNotFoundError(f"No LoRA adapter found under {sft_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the latest SFT LoRA adapter into Llama-3 weights.")
    parser.add_argument("--base_model", type=Path, default=Path("checkpoints/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--sft_dir", type=Path, default=Path("checkpoints/sft_checkpoint"))
    parser.add_argument("--adapter_dir", type=Path, default=None,
                        help="Specific adapter directory. Defaults to latest checkpoint-* under --sft_dir.")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/sft_merged"))
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device_map", default="auto",
                        help='Passed to from_pretrained when CUDA is available. Use "cpu" to force CPU.')
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--force", action="store_true", help="Overwrite output_dir if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model = args.base_model.resolve()
    sft_dir = args.sft_dir.resolve()
    adapter_dir = (args.adapter_dir.resolve() if args.adapter_dir else find_latest_adapter(sft_dir).resolve())
    output_dir = args.output_dir.resolve()

    if not base_model.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model}")
    if not has_adapter(adapter_dir):
        raise FileNotFoundError(f"Adapter files not found in: {adapter_dir}")
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output directory already exists: {output_dir} (pass --force to overwrite)")
        shutil.rmtree(output_dir)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if torch.cuda.is_available() and args.device_map != "cpu":
        model_kwargs["device_map"] = args.device_map

    print("[merge] base_model :", base_model)
    print("[merge] adapter    :", adapter_dir)
    print("[merge] output     :", output_dir)
    print("[merge] dtype      :", args.dtype)
    print("[merge] cuda       :", torch.cuda.is_available())

    base = AutoModelForCausalLM.from_pretrained(str(base_model), **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": str(base_model),
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "max_shard_size": args.max_shard_size,
        "created_at": datetime.utcnow().isoformat(),
    }
    with (output_dir / "sft_merge_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    del merged, model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[merge] done")


if __name__ == "__main__":
    main()
