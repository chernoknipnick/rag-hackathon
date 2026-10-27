"""Reranker (cross-encoder): точно переоценивает пары «вопрос + чанк». Бонус в ТЗ."""
import numpy as np

from .. import config
from ..schema import Hit
from .embedder import pick_device
from .index import index_text


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.RERANK_MODEL
        self.model = None

    def load(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=pick_device(),
                                      max_length=config.RERANK_MAX_LENGTH)
        return self

    def score(self, question: str, texts: list[str]) -> np.ndarray:
        """Вероятности 0–1. CrossEncoder с одним выходом по умолчанию применяет сигмоиду;
        если модель вернула сырые логиты (другая версия библиотеки), применяем её сами."""
        self.load()
        s = np.asarray(self.model.predict([(question, t) for t in texts], batch_size=16),
                       dtype="float64")
        if s.min() < 0 or s.max() > 1:
            s = 1.0 / (1.0 + np.exp(-s))
        return np.clip(s, 0.0, 1.0)


def rerank(reranker, question: str, hits: list[Hit]) -> list[Hit]:
    """Без реранкера оценкой становится косинус (у кандидатов только из BM25 — 0)."""
    if reranker is not None and hits:
        for h, s in zip(hits, reranker.score(question, [index_text(h.chunk) for h in hits])):
            h.score = float(s)
    else:
        for h in hits:
            h.score = max(h.dense or 0.0, 0.0)
    return sorted(hits, key=lambda h: h.score, reverse=True)
