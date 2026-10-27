"""Оценка качества: то, что жюри просит показать на питче.

Владелец: роль 4.

Запуск:
  python -m eval.evaluate eval/questions.json            # быстро: только поиск и роутер, без LLM
  python -m eval.evaluate eval/questions.json --full     # + LLM: отказы, факты в ответе, время
  python -m eval.evaluate eval/questions.json --full --out eval/results.json

Формат вопросов (все поля, кроме question, необязательны):
[
  {"question": "Какой класс точности у станка ТП-400?",
   "company": "tehnoprom",                       # для точности роутера
   "expected_doc": "tehnoprom/stanok_tp400.pdf", # подстрока пути правильного документа
   "expected_page": 12,                          # правильная страница
   "expected_contains": ["класс П"],             # факты, которые должны быть в чанке и в ответе
   "answerable": true}                           # false = правильный ответ «Нет в документах»
]

Метрики:
  router_acc      — доля вопросов, где роутер выбрал правильную компанию (или «все» включает её)
  hit@1, hit@3    — правильный чанк на 1-м месте / в первой тройке (hit rate из ТЗ)
  mrr@10          — средний 1/позиция первого правильного чанка
  refusal_acc     — доля вопросов без ответа, где система честно сказала «Нет в документах»
  false_refusals  — отказы на вопросы, у которых ответ есть
  fact_recall     — доля ожидаемых фактов, найденных в тексте ответа
  latency         — среднее, p50, p95 времени ответа, с
"""
import json
import statistics
import sys
import time

from src import config
from src.pipeline import RAG


def is_hit(item: dict, chunk) -> bool:
    doc, page, subs = item.get("expected_doc"), item.get("expected_page"), item.get("expected_contains", [])
    if doc and doc in chunk.doc and (page is None or chunk.page == page):
        return True
    return bool(subs) and any(s.lower() in chunk.text.lower() for s in subs)


def first_hit_rank(item: dict, chunks) -> int | None:
    for r, c in enumerate(chunks, 1):
        if is_hit(item, c):
            return r
    return None


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def evaluate(rag: RAG, items: list[dict], full: bool = False) -> dict:
    res = {"n": len(items)}
    router_ok = router_n = 0
    ranks, misses = [], []
    refuse_ok = refuse_n = false_refusals = 0
    facts_ok = facts_n = 0
    times = []

    for it in items:
        q, answerable = it["question"], it.get("answerable", True)
        route, hits, _ = rag.retrieve(q)
        if it.get("company"):
            router_n += 1
            router_ok += route.companies is None or it["company"] in route.companies
        if answerable and (it.get("expected_doc") or it.get("expected_contains")):
            r = first_hit_rank(it, [h.chunk for h in hits[:10]])
            ranks.append(r)
            if r is None or r > 3:
                misses.append({"question": q, "rank": r, "top3": [f"{h.chunk.doc} ({h.chunk.location})" for h in hits[:3]]})
        if full:
            rag.cache.clear()
            t0 = time.perf_counter()
            out = rag.ask(q)
            times.append(time.perf_counter() - t0)
            refused = out["answer"].startswith(config.NO_ANSWER)
            if not answerable:
                refuse_n += 1
                refuse_ok += refused
            else:
                false_refusals += refused
                for fact in it.get("expected_contains", []):
                    facts_n += 1
                    facts_ok += fact.lower() in out["answer"].lower()

    if router_n:
        res["router_acc"] = round(router_ok / router_n, 3)
    if ranks:
        res["hit@1"] = round(sum(r == 1 for r in ranks) / len(ranks), 3)
        res["hit@3"] = round(sum(r is not None and r <= 3 for r in ranks) / len(ranks), 3)
        res["mrr@10"] = round(sum(1 / r for r in ranks if r) / len(ranks), 3)
    if full:
        if refuse_n:
            res["refusal_acc"] = round(refuse_ok / refuse_n, 3)
        res["false_refusals"] = false_refusals
        if facts_n:
            res["fact_recall"] = round(facts_ok / facts_n, 3)
        if times:
            res["latency_mean_s"] = round(statistics.mean(times), 2)
            res["latency_p50_s"] = round(pct(times, 0.5), 2)
            res["latency_p95_s"] = round(pct(times, 0.95), 2)
    res["misses"] = misses
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "eval/questions.json"
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    items = json.load(open(path, encoding="utf-8"))
    res = evaluate(RAG(), items, full="--full" in sys.argv)

    print("| метрика | значение |\n|---|---|")
    for k, v in res.items():
        if k != "misses":
            print(f"| {k} | {v} |")
    if res["misses"]:
        print(f"\nПромахи hit@3 ({len(res['misses'])}):")
        for m in res["misses"]:
            print(f"- {m['question']}  ->  ранг {m['rank']}; топ-3: {m['top3']}")
    if out_path:
        json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
