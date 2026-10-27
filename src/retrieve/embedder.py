"""Эмбеддер (bge-m3 по умолчанию). Загружается один раз, лениво."""
import numpy as np

from .. import config


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Обёртка над SentenceTransformer. В тестах подменяется FakeEmbedder с тем же интерфейсом."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or config.EMBED_MODEL, device=pick_device())

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=16, normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


def embed(embedder, texts: list[str], prefix: str = "") -> np.ndarray:
    vecs = np.asarray(embedder.encode([prefix + t for t in texts]), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)
