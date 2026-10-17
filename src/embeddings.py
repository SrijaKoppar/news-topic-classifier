"""Sentence-embedding helpers, shared by training and the API."""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBED_BATCH_SIZE, EMBED_MAX_SEQ_LENGTH, EMBED_MODEL_NAME, RANDOM_SEED,
)


def load_encoder(name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """Load the pretrained encoder on CPU (downloads on first use)."""
    torch.manual_seed(RANDOM_SEED)
    model = SentenceTransformer(name, device="cpu")
    model.max_seq_length = EMBED_MAX_SEQ_LENGTH
    return model


def encode_texts(model: SentenceTransformer, texts, show_progress: bool = True) -> np.ndarray:
    """Texts -> (n, 384) float32 array of unit-length embeddings."""
    emb = model.encode(
        list(texts),
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)