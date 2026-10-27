"""Пайплайн: эмбеддинг вопроса -> роутер -> гибридный поиск -> reranker -> порог -> LLM -> ответ.

Владелец: роль 5 (сборка). Каждый шаг живёт в модуле своего владельца,
здесь только склейка и замер времени по шагам.
"""
import time
from collections import OrderedDict

from . import config
from .generate.answer import build_answer, refusal
from .generate.llm import generate
from .retrieve.hybrid import hybrid_search
from .retrieve.index import Index
from .retrieve.rerank import Reranker, rerank
from .router import Router
from .schema import Hit, Route


class RAG:
    def __init__(self, index: Index | None = None, reranker="auto", llm=None, router="auto"):
        """reranker / router: "auto" (по настройкам), None (выключен) или свой объект."""
        self.index = index or Index()
        if reranker == "auto":
            reranker = Reranker() if config.USE_RERANKER else None
        self.reranker = reranker
        self.llm = llm or generate
        self._auto_router = router == "auto"
        self.router = Router.from_index(self.index) if self._auto_router else (router or Router())
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def on_index_changed(self):
        """После индексации или /update: пересчитать центроиды роутера и сбросить кэш."""
        if self._auto_router:
            self.router = Router.from_index(self.index)
        self.cache.clear()

    def _search(self, question, qvec, companies) -> list[Hit]:
        hits = hybrid_search(self.index, question, qvec, companies, config.TOP_K_RETRIEVE)
        return rerank(self.reranker, question, hits)

    def retrieve(self, question: str) -> tuple[Route, list[Hit], dict]:
        t = {}
        t0 = time.perf_counter()
        qvec = self.index.embed_query(question)
        t["embed_ms"] = (t1 := time.perf_counter()) - t0
        route = self.router.route(question, qvec)
        t["route_ms"] = (t2 := time.perf_counter()) - t1
        hits = self._search(question, qvec, route.companies)
        # Agentic-шаг: в выбранной компании ничего уверенного -> ищем по всем компаниям
        if config.ROUTER_FALLBACK_ALL and route.companies and (
                not hits or hits[0].score < config.MIN_CONFIDENCE):
            wide = self._search(question, qvec, None)
            if wide and (not hits or wide[0].score > hits[0].score):
                hits, route = wide, Route(None, route.method + "->all", route.scores)
        t["search_ms"] = time.perf_counter() - t2
        return route, hits, t

    def ask(self, question: str) -> dict:
        key = " ".join(question.lower().split())
        if key in self.cache:  # кэш частых вопросов
            self.cache.move_to_end(key)
            return {**self.cache[key], "cached": True}

        t0 = time.perf_counter()
        route, hits, timings = self.retrieve(question)
        confidence = hits[0].score if hits else 0.0
        ctx = hits[: config.TOP_K_CONTEXT]
        cacheable = True

        if not hits or confidence < config.MIN_CONFIDENCE:
            result = {**refusal(confidence), "unsure": True}  # отказ без вызова LLM
        else:
            t1 = time.perf_counter()
            try:
                text = self.llm(question, ctx)
            except Exception:  # LLM недоступна -> отдаём лучший фрагмент, а не падаем
                text, cacheable = f"(LLM недоступна, фрагмент документа) {ctx[0].chunk.text[:400]} [1]", False
            timings["llm_ms"] = time.perf_counter() - t1
            result = build_answer(text, ctx, confidence)

        timings["total_ms"] = time.perf_counter() - t0
        result.update({
            "company": route.companies or "all",
            "route": route.method,
            "latency_ms": int(timings["total_ms"] * 1000),
            "timings_ms": {k: int(v * 1000) for k, v in timings.items()},
        })
        if cacheable:
            self.cache[key] = result
            if len(self.cache) > config.CACHE_SIZE:
                self.cache.popitem(last=False)
        return result

    def warmup(self):
        """Прогрев: эмбеддер, reranker и LLM загружаются при старте, а не на первом вопросе."""
        from .generate.llm import warmup as llm_warmup
        self.index.embed_query("прогрев")
        if self.reranker is not None and hasattr(self.reranker, "load"):
            self.reranker.load()
        if self.llm is generate:
            llm_warmup()
