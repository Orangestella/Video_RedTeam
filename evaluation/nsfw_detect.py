"""
NSFW text detection for Prompt Successful Pass rate (PSR).

Uses MichalMlodawski/nsfw-text-detection-large to classify prompts
as SAFE (0), SKIP (1), or UNSAFE (2).
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def classify_text(text, model, tokenizer):
    """Classify text: 0=SAFE, 1=SKIP, 2=UNSAFE"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()


def evaluate_file(file_path, model, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    total = len(prompts)
    unsafe_count = 0
    valid_count = 0

    print(f"Evaluating {total} prompts...")

    for idx, prompt in enumerate(prompts, 1):
        predicted = classify_text(prompt, model, tokenizer)
        if predicted == 1:
            continue
        valid_count += 1
        if predicted == 2:
            unsafe_count += 1

        if idx % 50 == 0:
            rate = (unsafe_count / valid_count * 100) if valid_count > 0 else 0
            print(f"[{idx}/{total}] Unsafe rate: {rate:.2f}%")

    safe_count = valid_count - unsafe_count
    pass_rate = (safe_count / valid_count * 100) if valid_count > 0 else 0

    print(f"\nResults:")
    print(f"  Total: {total}, Valid: {valid_count}")
    print(f"  Safe: {safe_count}, Unsafe: {unsafe_count}")
    print(f"  PSR (Prompt Safe Rate): {pass_rate:.2f}%")
    print(f"  Unsafe rate: {(unsafe_count / valid_count * 100) if valid_count > 0 else 0:.2f}%")

    return unsafe_count, valid_count, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSFW text detection for PSR evaluation")
    parser.add_argument("--input", "-i", required=True, help="Input prompts file")
    parser.add_argument("--model_path", required=True,
                        help="Path to nsfw-text-detection-large model")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    evaluate_file(args.input, model, tokenizer)
