"""
Embedding cosine similarity diversity metric (1-Cossim).

Computes average pairwise cosine similarity using text embeddings.
Paper metric: 1-Cossim (reported in Table 2).
"""

import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def read_corpus_from_file(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_embeddings_batch(texts: list, client, model="text-embedding-ada-002", batch_size=100) -> list:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch, encoding_format="float")
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Batch {i // batch_size + 1} failed: {e}, trying one by one...")
            for text in batch:
                try:
                    resp = client.embeddings.create(model=model, input=text, encoding_format="float")
                    embeddings.append(resp.data[0].embedding)
                except Exception:
                    pass
    return embeddings


def calculate_average_cosine_similarity(embeddings: list) -> dict:
    embeddings_array = np.array(embeddings)
    sim_matrix = cosine_similarity(embeddings_array)
    n = len(embeddings)
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    similarities = sim_matrix[mask]

    return {
        'average': float(np.mean(similarities)),
        'max': float(np.max(similarities)),
        'min': float(np.min(similarities)),
        'std': float(np.std(similarities)),
        'median': float(np.median(similarities)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embedding cosine similarity")
    parser.add_argument("--input", "-i", required=True, help="Input file (one prompt per line)")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--base_url", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--model", default="text-embedding-ada-002", help="Embedding model")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    corpus = read_corpus_from_file(args.input)
    print(f"Loaded {len(corpus)} sentences")

    embeddings = get_embeddings_batch(corpus, client, model=args.model)
    print(f"Got {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    stats = calculate_average_cosine_similarity(embeddings)

    print(f"\nAvg cosine similarity: {stats['average']:.4f}")
    print(f"1-Cossim (diversity): {1 - stats['average']:.4f}")
    print(f"Max: {stats['max']:.4f}, Min: {stats['min']:.4f}, Std: {stats['std']:.4f}")
