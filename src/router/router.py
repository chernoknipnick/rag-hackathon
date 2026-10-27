"""Роутер по компании (требование 5 ТЗ, бонус A).

Владелец: роль 4.

Порядок решения:
1. Правила (ключевые слова). Один явный победитель -> только эта компания.
2. Ничья по правилам -> NCM выбирает среди спорных компаний, если уверен.
3. Правила молчат -> NCM по всем компаниям, если разрыв 1-го и 2-го ≥ ROUTER_NCM_MARGIN.
4. Иначе -> поиск по всем компаниям (сквозной вопрос).
Пайплайн дополнительно повторяет поиск по всем компаниям, если в выбранной
ничего уверенного не нашлось (ROUTER_FALLBACK_ALL) — защита от ошибки роутера.
"""
import numpy as np

from .. import config
from ..schema import Route
from .ncm import NCM
from .rules import rule_scores


class Router:
    def __init__(self, ncm: NCM | None = None, margin: float | None = None):
        self.ncm = ncm
        self.margin = config.ROUTER_NCM_MARGIN if margin is None else margin

    @classmethod
    def from_index(cls, index) -> "Router":
        vecs = index.company_vectors()
        return cls(NCM(config.ROUTER_CENTROIDS).fit(vecs) if vecs else None)

    def _ncm_pick(self, qvec, candidates: list[str]) -> tuple[str | None, dict[str, float]]:
        if self.ncm is None or qvec is None:
            return None, {}
        scores = {c: s for c, s in self.ncm.scores(qvec).items() if c in candidates}
        if len(scores) < 2:
            return (next(iter(scores)) if scores else None), scores
        ranked = sorted(scores.values(), reverse=True)
        best = max(scores, key=scores.get)
        return (best if ranked[0] - ranked[1] >= self.margin else None), scores

    def route(self, question: str, qvec: np.ndarray | None = None) -> Route:
        rules = rule_scores(question)
        top = max(rules.values(), default=0)
        if top > 0:
            leaders = [c for c, s in rules.items() if s == top]
            if len(leaders) == 1:
                return Route(leaders, "rules", rules)
            pick, scores = self._ncm_pick(qvec, leaders)
            if pick:
                return Route([pick], "rules+ncm", scores)
            return Route(leaders if len(leaders) < len(rules) else None, "rules", rules)
        pick, scores = self._ncm_pick(qvec, list(config.COMPANY_TITLES))
        if pick:
            return Route([pick], "ncm", scores)
        return Route(None, "all", scores)
