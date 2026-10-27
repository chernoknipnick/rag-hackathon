"""Сборка ответа: порог отказа, источники по цитатам, confidence, самопроверка чисел.

Владелец: роль 3 (генерация).
"""
import re

from .. import config
from ..schema import Hit

_CITE = re.compile(r"\[(\d{1,2})\]")
_NUM = re.compile(r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?")


def is_refusal(text: str) -> bool:
    return config.NO_ANSWER.lower() in text.lower()[:80]


def refusal(confidence: float) -> dict:
    return {"answer": config.NO_ANSWER, "sources": [], "confidence": round(confidence, 3)}


def _norm_num(s: str) -> str:
    return s.replace(" ", "").replace(" ", "").replace(",", ".")


def _num_forms(n: str) -> set[str]:
    """«12 000» -> {"12000", "12", "000"}: и целиком, и по частям (в тексте числа идут подряд)."""
    return {_norm_num(n)} | {_norm_num(p) for p in re.split(r"[  ]", n) if p}


def unsupported_numbers(answer: str, hits: list[Hit]) -> list[str]:
    """Числа из ответа, которых нет ни в одном фрагменте контекста (признак галлюцинации).
    Однозначные числа не проверяем: в тексте они часто написаны словами («один раз»)."""
    context_nums: set[str] = set()
    for h in hits:
        for n in _NUM.findall(h.chunk.text + " " + h.chunk.location + " " + h.chunk.section):
            context_nums |= _num_forms(n)
    out = []
    for n in _NUM.findall(_CITE.sub("", answer)):
        if len(_norm_num(n)) > 1 and _norm_num(n) not in context_nums:
            out.append(n)
    return out


def cited_hits(answer: str, hits: list[Hit]) -> list[Hit]:
    nums = [int(n) for n in _CITE.findall(answer)]
    return [hits[n - 1] for n in dict.fromkeys(nums) if 1 <= n <= len(hits)]


def build_answer(answer: str, hits: list[Hit], confidence: float) -> dict:
    """answer — текст LLM; hits — фрагменты, которые были в контексте (в том же порядке)."""
    if is_refusal(answer):
        return {**refusal(min(confidence, 0.2)), "unsure": True}

    unsupported = unsupported_numbers(answer, hits)
    unsure = confidence < config.UNSURE_CONFIDENCE or bool(unsupported)
    if unsupported:  # число не подтверждается документами -> снижаем уверенность
        confidence *= 0.5

    used = cited_hits(answer, hits) or hits[:1]
    seen, sources = set(), []
    for h in used[:3]:
        key = (h.chunk.doc, h.chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append({"doc": h.chunk.doc, "page": h.chunk.page, "score": round(h.score, 3),
                            "location": h.chunk.location})
    text = re.sub(r"\s+([.,;:])", r"\1", _CITE.sub("", answer)).strip()
    return {"answer": text, "sources": sources, "confidence": round(confidence, 3),
            "unsure": unsure, "unsupported_numbers": unsupported}
