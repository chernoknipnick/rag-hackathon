"""Контракты между модулями. Меняется только по договорённости всей команды.

Поток данных:
    ingest  -> list[Chunk]
    retrieve: search(...) -> list[Hit]
    router:   route(...) -> Route
    generate: answer(...) -> dict в формате ответа /ask
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    id: str
    text: str             # текст фрагмента (то, что видит LLM)
    doc: str              # путь относительно docs/, например tehnoprom/stanok_tp400.pdf
    company: str          # первая папка в docs/
    page: int | None      # номер страницы PDF; для остальных форматов: номер абзаца / строки / записи
    location: str         # подпись для человека: «стр. 12», «абзацы 3–9», «Тарифы, строка 7»
    section: str = ""     # заголовок раздела / лист таблицы
    kind: str = "text"    # text | table_row | record | image

    def header(self, title: str = "") -> str:
        """Шапка чанка: добавляется к тексту при индексации, улучшает поиск."""
        parts = [title or self.company, self.doc]
        if self.section:
            parts.append(self.section)
        return "[" + " · ".join(parts) + "]"

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    chunk: Chunk
    score: float                  # итоговая оценка: вероятность реранкера (0–1) или косинус
    dense: float | None = None    # косинус из векторного поиска
    sparse: float | None = None   # скор BM25


@dataclass
class Route:
    companies: list[str] | None   # None = искать по всем (сквозной вопрос или не определили)
    method: str                   # rules | rules+ncm | ncm | all
    scores: dict[str, float] = field(default_factory=dict)


# Формат ответа POST /ask строго по ТЗ (раздел 6). Остальные поля только в debug.
STRICT_KEYS = ("answer", "sources", "confidence")
SOURCE_KEYS = ("doc", "page", "score")


def strict_response(result: dict) -> dict:
    return {
        "answer": result["answer"],
        "sources": [{k: s[k] for k in SOURCE_KEYS} for s in result["sources"]],
        "confidence": result["confidence"],
    }
