"""NCM (Nearest Class Mean): класс = ближайший центроид.

Центроиды считаются по векторам документов каждой компании (метка компании
бесплатно известна из папки). Чтобы одна компания с разнородными документами
(карточки товаров, политика возврата, чаты) не схлопывалась в одну точку,
на компанию строится несколько центроидов (k-means), скор = ближайший из них.
Дополнительно можно подмешать эмбеддинги синтетических вопросов (add_examples).
"""
import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1, n)


def kmeans(x: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> np.ndarray:
    """Сферический k-means (косинус). Детерминированный: одинаковый результат при каждом запуске."""
    k = max(1, min(k, len(x)))
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(k - 1):  # k-means++: следующий центр подальше от уже выбранных
        d = 1 - np.max(x @ np.stack(centers).T, axis=1)
        p = np.clip(d, 0, None)
        centers.append(x[rng.choice(len(x), p=p / p.sum())] if p.sum() > 0 else x[rng.integers(len(x))])
    c = np.stack(centers)
    for _ in range(iters):
        labels = np.argmax(x @ c.T, axis=1)
        new = np.stack([x[labels == j].mean(axis=0) if np.any(labels == j) else c[j] for j in range(k)])
        c = _normalize(new)
    return c


class NCM:
    def __init__(self, n_centroids: int = 3):
        self.n_centroids = n_centroids
        self.centroids: dict[str, np.ndarray] = {}

    def fit(self, vectors_by_company: dict[str, np.ndarray]) -> "NCM":
        self.centroids = {c: kmeans(_normalize(v), self.n_centroids)
                          for c, v in vectors_by_company.items() if len(v)}
        return self

    def add_examples(self, company: str, vecs: np.ndarray):
        """Добавить центроид из примеров вопросов (например, синтетических)."""
        mean = _normalize(np.asarray(vecs).mean(axis=0, keepdims=True))
        prev = self.centroids.get(company)
        self.centroids[company] = mean if prev is None else np.vstack([prev, mean])

    def scores(self, qvec: np.ndarray) -> dict[str, float]:
        return {c: float(np.max(cents @ qvec)) for c, cents in self.centroids.items()}
