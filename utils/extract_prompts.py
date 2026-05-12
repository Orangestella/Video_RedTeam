"""
Extract generated prompts from TEAR pipeline logs.

Supports two modes:
  --mode all    : Extract all final prompts from the log.
  --mode best   : Extract the highest-scoring safe prompt per seed from all iterations.
"""

import argparse
import json
import sys
from pathlib import Path


def extract_all(log_data):
    prompts = []
    for entry in log_data.get('prompts', []):
        final = entry.get('final_prompt')
        if final:
            prompts.append(final)
    return prompts


def extract_best_safe(log_data):
    prompts = []
    for entry in log_data.get('prompts', []):
        iterations = entry.get('iterations', [])
        if not iterations:
            final = entry.get('final_prompt')
            if final:
                prompts.append(final)
            continue

        best_prompt = None
        best_score = -1
        for it in iterations:
            guard = it.get('guard_result', '')
            if guard and 'unsafe' in guard.lower():
                continue
            score_str = it.get('score')
            if score_str and str(score_str).isdigit():
                score = int(score_str)
                if score > best_score:
                    best_score = score
                    best_prompt = it.get('prompt', '')

        if best_prompt is None:
            final = entry.get('final_prompt')
            if final:
                prompts.append(final)
        else:
            prompts.append(best_prompt)

    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prompts from TEAR pipeline log")
    parser.add_argument("--log", "-l", required=True, help="Path to pipeline_log.json")
    parser.add_argument("--output", "-o", required=True, help="Output file (one prompt per line)")
    parser.add_argument("--mode", choices=["all", "best"], default="all",
                        help="'all': extract final prompts; 'best': highest-scoring safe prompt per seed")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: log file not found: {log_path}")
        sys.exit(1)

    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    if args.mode == "best":
        prompts = extract_best_safe(log_data)
    else:
        prompts = extract_all(log_data)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(f"{p}\n")

    print(f"Extracted {len(prompts)} prompts ({args.mode} mode) -> {args.output}")
