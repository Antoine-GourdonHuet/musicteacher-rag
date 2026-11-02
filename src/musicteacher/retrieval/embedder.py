"""Dense embedding model (BAAI/bge-m3), configured the same way for indexing and for queries."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Our chunks are at most ~450 tokens + a title line; 512 keeps memory low on an 8 GB GPU.
MAX_SEQ_LENGTH = 512


def load_embedder(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()  # float16: half the memory, faster, negligible quality loss for retrieval
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """L2-normalised vectors, so cosine similarity = dot product."""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
    )
