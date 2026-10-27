"""Роли 3 и 5: формат ответа ТЗ, отказ, источники по цитатам, самопроверка, кэш, роутер-фолбэк."""
from src import config
from src.generate.answer import build_answer, unsupported_numbers
from src.pipeline import RAG
from src.router import Router
from src.schema import Chunk, Hit, Route, strict_response


def make_rag(index, answer="Штраф 0,1% от стоимости груза за каждый день [1].", **kw):
    return RAG(index=index, reranker=None, llm=lambda q, hits: answer, **kw)


def test_strict_response_matches_tz(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.05)
    res = strict_response(make_rag(index).ask("Есть ли в договоре с ТрансГруз штраф за просрочку доставки?"))
    assert set(res) == {"answer", "sources", "confidence"}
    assert set(res["sources"][0]) == {"doc", "page", "score"}
    assert res["sources"][0]["doc"] == "logistics/dogovor_transgruz.docx"
    assert isinstance(res["sources"][0]["page"], int)
    assert 0 <= res["confidence"] <= 1 and "[1]" not in res["answer"]


def test_refusal_without_llm_call(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.5)
    called = []
    rag = RAG(index=index, reranker=None, llm=lambda q, h: called.append(1) or "x")
    res = rag.ask("zzzz qqqq xxxx")
    assert res["answer"] == config.NO_ANSWER and res["sources"] == [] and not called


def test_llm_refusal_drops_sources(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.05)
    res = make_rag(index, answer="Нет в документах.").ask("штраф за просрочку доставки")
    assert res["sources"] == [] and res["confidence"] <= 0.2


def test_cache(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.05)
    rag = make_rag(index)
    rag.ask("штраф за просрочку доставки")
    assert rag.ask("Штраф  за просрочку доставки").get("cached") is True
    rag.on_index_changed()
    assert "cached" not in rag.ask("штраф за просрочку доставки")


def test_timings_reported(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.05)
    res = make_rag(index).ask("штраф за просрочку доставки")
    assert {"embed_ms", "route_ms", "search_ms", "llm_ms", "total_ms"} <= set(res["timings_ms"])


def _hit(text, doc="a.pdf", page=1, score=0.9):
    return Hit(Chunk(f"id-{doc}-{page}", text, doc, "x", page, f"стр. {page}"), score)


def test_sources_follow_citations():
    hits = [_hit("масса 1200 кг", "a.pdf", 2), _hit("класс точности П", "b.pdf", 7)]
    res = build_answer("Класс точности П [2].", hits, 0.9)
    assert [s["doc"] for s in res["sources"]] == ["b.pdf"] and res["sources"][0]["page"] == 7
    assert res["answer"] == "Класс точности П."


def test_unsupported_number_lowers_confidence():
    hits = [_hit("Тариф Москва — Казань: 12 400 руб за 500 кг")]
    assert unsupported_numbers("Стоимость 12400 руб [1]", hits) == []
    assert unsupported_numbers("Стоимость 15 000 руб [1]", hits) == ["15 000"]
    res = build_answer("Стоимость 15 000 руб [1]", hits, 0.8)
    assert res["confidence"] == 0.4 and res["unsure"] is True


def test_wrong_route_falls_back_to_all_companies(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.2)

    class WrongRouter(Router):
        def route(self, question, qvec=None):
            return Route(["tehnoprom"], "rules")

    rag = make_rag(index, router=WrongRouter())
    route, hits, _ = rag.retrieve("Остаток полетный контроллер FC-1")
    assert route.companies is None and route.method.endswith("->all")
    assert hits[0].chunk.company == "retail"


def test_llm_down_returns_fragment(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.05)

    def boom(q, h):
        raise ConnectionError

    res = RAG(index=index, reranker=None, llm=boom).ask("штраф за просрочку доставки")
    assert res["answer"].startswith("(LLM недоступна") and res["sources"]
