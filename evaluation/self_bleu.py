"""
Self-BLEU diversity metric (1-AvgSelfBLEU).

Lower Self-BLEU indicates higher prompt diversity.
Paper metric: 1-AvgSelfBLEU (reported in Table 2).
"""

import argparse
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def simple_tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())


def bleu(hypothesis: str, references: list) -> float:
    hypothesis_tokens = simple_tokenize(hypothesis)
    references_tokens = [simple_tokenize(ref) for ref in references]
    smoothing = SmoothingFunction().method1
    return sentence_bleu(references_tokens, hypothesis_tokens, smoothing_function=smoothing)


def get_self_bleu_score(corpus: list) -> float:
    if len(corpus) < 2:
        return 0.0

    score = 0.0
    for index in range(len(corpus)):
        curr_text = corpus[index]
        other_text = corpus[:index] + corpus[index + 1:]
        score += bleu(curr_text, other_text)

        if (index + 1) % 50 == 0:
            print(f"Processed: {index + 1}/{len(corpus)}")

    return score / len(corpus)


def read_corpus_from_file(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Self-BLEU diversity score")
    parser.add_argument("--input", "-i", required=True, help="Input file (one prompt per line)")
    args = parser.parse_args()

    corpus = read_corpus_from_file(args.input)
    print(f"Loaded {len(corpus)} sentences")

    self_bleu = get_self_bleu_score(corpus)
    diversity = 1 - self_bleu

    print(f"\nSelf-BLEU: {self_bleu:.4f}")
    print(f"1-AvgSelfBLEU (diversity): {diversity:.4f}")
