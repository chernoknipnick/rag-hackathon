"""Роль 4: правила, NCM и их сочетание."""
import numpy as np

from src.router import Router
from src.router.ncm import NCM, kmeans
from src.router.rules import rule_scores


def test_rules_match_word_start_only():
    assert rule_scores("Как начать работу?")["retail"] == 0  # «чат» внутри «начать» не считается
    assert rule_scores("Покажи чаты поддержки")["retail"] >= 1


def test_rules_route_examples_from_tz():
    r = Router()
    assert r.route("Сколько стоит доставка груза 500 кг из Москвы в Казань?").companies == ["logistics"]
    assert r.route("Как вернуть полетный контроллер, если прошло 20 дней?").companies == ["retail"]
    assert r.route("Какой класс точности у станка модели ТП-400?").companies == ["tehnoprom"]
    assert r.route("Есть ли в договоре с «ТрансГруз» пункт о штрафах?").companies == ["logistics"]


def test_no_signal_searches_everywhere():
    route = Router().route("Какая погода в Твери?")
    assert route.companies is None and route.method == "all"


def _ncm():
    e = np.eye(4, dtype="float32")
    return NCM(1).fit({"tehnoprom": e[[0]], "logistics": e[[1]], "retail": e[[2]]})


def test_ncm_decides_when_rules_silent():
    route = Router(_ncm(), margin=0.1).route("непонятный вопрос", np.array([0.1, 0.9, 0.1, 0], "float32"))
    assert route.companies == ["logistics"] and route.method == "ncm"


def test_ncm_breaks_rule_tie():
    q = "договор на возврат"  # правила: logistics («договор») и retail («возврат») поровну
    assert len(Router().route(q).companies) == 2
    route = Router(_ncm(), margin=0.1).route(q, np.array([0, 0.2, 0.9, 0], "float32"))
    assert route.companies == ["retail"] and route.method == "rules+ncm"


def test_ncm_small_margin_means_all():
    route = Router(_ncm(), margin=0.1).route("непонятный вопрос", np.array([0.5, 0.5, 0.5, 0], "float32"))
    assert route.companies is None


def test_kmeans_separates_clusters():
    rng = np.random.default_rng(1)
    a = rng.normal([1, 0, 0], 0.05, (20, 3))
    b = rng.normal([0, 1, 0], 0.05, (20, 3))
    x = np.vstack([a, b])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    c = kmeans(x, 2)
    assert sorted(np.argmax(np.abs(c), axis=1).tolist()) == [0, 1]


def test_router_from_index_uses_document_centroids(index):
    router = Router.from_index(index)
    assert set(router.ncm.centroids) == {"tehnoprom", "logistics", "retail"}
