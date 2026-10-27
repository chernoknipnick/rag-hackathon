"""Индекс: чанки + векторы (numpy) + BM25. Хранится в INDEX_DIR.

Владелец: роль 2 (поиск).

Файлы индекса:
  chunks.jsonl     — чанки с метаданными
  vectors.npy      — нормализованные эмбеддинги (строки в порядке vector_ids.txt)
  vector_ids.txt
Индекс можно собрать заранее (на GPU) и отдать жюри готовым — см. README.
"""
import json
import time
from pathlib import Path

import numpy as np

from .. import config
from ..ingest.parsers import PARSERS, parse_file, rel_path
from ..schema import Chunk
from .bm25 import BM25, tokenize
from .embedder import Embedder, embed
from .vector_store import VectorStore

SAVE_EVERY = 50  # файлов; при обрыве индексация продолжится с места остановки


def index_text(c: Chunk) -> str:
    """То, что попадает в эмбеддинг и BM25: шапка (компания · файл · раздел) + текст."""
    return c.header(config.COMPANY_TITLES.get(c.company, c.company)) + "\n" + c.text


class Index:
    def __init__(self, embedder=None, index_dir: Path | None = None):
        self.dir = Path(index_dir or config.INDEX_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()
        self.dim = int(getattr(self.embedder, "dim", 0) or self.embedder.encode(["x"]).shape[1])
        self.chunks: dict[str, Chunk] = {}
        self._load_chunks()
        self.vectors = VectorStore.load(self.dir, self.dim)
        if set(self.vectors.ids) != set(self.chunks):  # рассинхрон -> пересчитать векторы
            self.vectors = VectorStore(self.dim)
            self._embed_and_add(list(self.chunks.values()))
        self._rebuild()

    # ---------- хранение ----------
    def _load_chunks(self):
        f = self.dir / "chunks.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c = Chunk(**json.loads(line))
                    self.chunks[c.id] = c

    def save(self):
        with open(self.dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks.values():
                f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
        self.vectors.save(self.dir)

    def _rebuild(self):
        """BM25 и маски компаний пересчитываются после изменений (секунды на 20 тыс. чанков)."""
        self._order = list(self.chunks)
        self._bm25 = BM25().fit([tokenize(index_text(self.chunks[i])) for i in self._order])
        self._vec_company = np.asarray([self.chunks[i].company for i in self.vectors.ids])

    def _embed_and_add(self, chunks: list[Chunk]):
        for i in range(0, len(chunks), 256):
            batch = chunks[i:i + 256]
            vecs = embed(self.embedder, [index_text(c) for c in batch], config.PASSAGE_PREFIX)
            self.vectors.add([c.id for c in batch], vecs)

    # ---------- индексация ----------
    def remove_doc(self, doc: str):
        drop = {cid for cid, c in self.chunks.items() if c.doc == doc}
        for cid in drop:
            del self.chunks[cid]
        self.vectors.remove(drop)

    def add_chunks(self, chunks: list[Chunk], rebuild: bool = True) -> int:
        for doc in {c.doc for c in chunks}:  # переиндексация документа = удалить старое, добавить новое
            self.remove_doc(doc)
        chunks = list({c.id: c for c in chunks}.values())
        self._embed_and_add(chunks)
        for c in chunks:
            self.chunks[c.id] = c
        if rebuild:
            self._rebuild()
            self.save()
        return len(chunks)

    def ingest_file(self, path, root: Path | None = None) -> int:
        """Инкрементальная индексация одного файла (POST /update, бонус B)."""
        return self.add_chunks(parse_file(path, root))

    def ingest_dir(self, root: Path | None = None, force: bool = False) -> int:
        root = Path(root or config.DOCS_DIR)
        known = {c.doc for c in self.chunks.values()}
        total, done, t0 = 0, 0, time.perf_counter()
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in PARSERS:
                continue
            rel = rel_path(p, root)
            if not force and rel in known:
                continue
            try:
                n = self.add_chunks(parse_file(p, root), rebuild=False)
                total += n
                done += 1
                print(f"[ingest] {rel}: {n} чанков", flush=True)
            except Exception as e:  # один битый файл не должен ронять всё
                print(f"[ingest] ОШИБКА {rel}: {e}", flush=True)
            if done and done % SAVE_EVERY == 0:
                self.save()
        self._rebuild()
        self.save()
        print(f"[ingest] файлов: {done}, чанков: {total}, {time.perf_counter() - t0:.1f} с", flush=True)
        return total

    # ---------- поиск ----------
    def embed_query(self, question: str) -> np.ndarray:
        return embed(self.embedder, [question], config.QUERY_PREFIX)[0]

    def dense_search(self, qvec: np.ndarray, companies: list[str] | None, k: int):
        mask = np.isin(self._vec_company, companies) if companies else None
        return self.vectors.search(qvec, k, mask)

    def bm25_search(self, question: str, companies: list[str] | None, k: int):
        scores = self._bm25.scores(tokenize(question))
        out = []
        for i in np.argsort(-scores):
            if scores[i] <= 0:
                break
            cid = self._order[i]
            if companies and self.chunks[cid].company not in companies:
                continue
            out.append((cid, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def company_vectors(self) -> dict[str, np.ndarray]:
        """Все векторы, сгруппированные по компании (для роутера NCM)."""
        return {c: self.vectors.vecs[self._vec_company == c]
                for c in np.unique(self._vec_company)} if len(self.vectors) else {}
