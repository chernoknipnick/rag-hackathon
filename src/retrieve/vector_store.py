"""Векторное хранилище на numpy: точный поиск по косинусу.

Почему не Qdrant: на ~500 документах (10–20 тыс. чанков) полный перебор
матрицей занимает миллисекунды на CPU, нет блокировки «один процесс»,
нет отдельного сервиса в Docker. Если понадобится масштаб, интерфейс
(add / remove / search / save / load) легко переложить на FAISS или Qdrant.
"""
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.vecs = np.zeros((0, dim), dtype="float32")

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vecs: np.ndarray):
        self.ids += ids
        self.vecs = np.vstack([self.vecs, np.asarray(vecs, dtype="float32")])

    def remove(self, drop: set[str]):
        if not drop:
            return
        keep =[i for i, cid in enumerate(self.ids) if cid not in drop]
        self.ids = [self.ids[i] for i in keep]
        self.vecs = self.vecs[keep]

    def search(self, qvec: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.vecs @ qvec
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top if np.isfinite(sims[i])]

    def vectors_of(self, rows: np.ndarray) -> np.ndarray:
        return self.vecs[rows]

    def save(self, folder: Path):
        np.save(folder / "vectors.npy", self.vecs)
        (folder / "vector_ids.txt").write_text("\n".join(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, dim: int) -> "VectorStore":
        store = cls(dim)
        f, ids = folder / "vectors.npy", folder / "vector_ids.txt"
        if f.exists() and ids.exists():
            store.vecs = np.load(f).astype("float32")
            store.ids = [x for x in ids.read_text(encoding="utf-8").split("\n") if x]
            if len(store.ids) != len(store.vecs) or store.vecs.shape[1] != dim:
                return cls(dim)  # индекс от другой модели или битый -> начать заново
        return store
