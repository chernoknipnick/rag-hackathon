"""Роль 2: BM25, индекс, гибридный поиск, инкрементальное обновление."""
from src.retrieve.bm25 import BM25, tokenize
from src.retrieve.hybrid import hybrid_search, rrf
from src.retrieve.index import Index

from .conftest import FakeEmbedder


def test_tokenize_model_codes_cyrillic_equals_latin():
    assert "tp-400" in tokenize("станок ТП-400") and "tp-400" in tokenize("stanok TP-400")
    assert "400" in tokenize("ТП 400")
    assert tokenize("станками")[0] == tokenize("станком")[0] == "станк"
    assert len({tokenize(w)[0] for w in ["доставка", "доставки", "доставку", "доставкой"]}) == 1
    assert tokenize("штрафах")[0] == tokenize("штраф")[0]


def test_bm25_ranks_relevant_first():
    docs = [tokenize(t) for t in ["доставка груза в Казань", "возврат товара", "класс точности станка"]]
    s = BM25().fit(docs).scores(tokenize("доставка в Казань"))
    assert s.argmax() == 0 and s[1] == 0


def test_rrf_rewards_agreement():
    fused = rrf([[("a", 1), ("b", 1)], [("b", 1), ("c", 1)]])
    assert max(fused, key=fused.get) == "b"


def test_company_filter(index):
    q = "штраф за просрочку доставки"
    hits = hybrid_search(index, q, index.embed_query(q), ["retail"], 10)
    assert hits and all(h.chunk.company == "retail" for h in hits)
    hits = hybrid_search(index, q, index.embed_query(q), ["logistics"], 10)
    assert hits[0].chunk.doc == "logistics/dogovor_transgruz.docx"


def test_persistence_and_incremental_update(index, docs, tmp_path):
    n = len(index.chunks)
    reloaded = Index(embedder=FakeEmbedder(), index_dir=tmp_path / "index")
    assert len(reloaded.chunks) == n and len(reloaded.vectors) == n

    new = docs / "logistics" / "reglament.txt"
    new.write_text("Новый регламент склада: приёмка грузов с 9 до 18.", encoding="utf-8")
    assert reloaded.ingest_file(new, docs) == 1
    q = "регламент склада приёмка"
    assert hybrid_search(reloaded, q, reloaded.embed_query(q), None, 3)[0].chunk.doc == "logistics/reglament.txt"

    new.write_text("Регламент склада отменён.", encoding="utf-8")  # переиндексация: старое удаляется
    reloaded.ingest_file(new, docs)
    texts = [c.text for c in reloaded.chunks.values() if c.doc == "logistics/reglament.txt"]
    assert texts == ["Регламент склада отменён."]
    assert len(reloaded.vectors) == len(reloaded.chunks)


def test_resume_skips_known_files(index, docs):
    assert index.ingest_dir(docs) == 0
