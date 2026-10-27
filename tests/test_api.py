"""Роль 5: HTTP API. Пропускается, если не установлены fastapi/httpx."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src import api, config  # noqa: E402
from src.pipeline import RAG  # noqa: E402


@pytest.fixture()
def client(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.05)
    api.rag = RAG(index=index, reranker=None, llm=lambda q, h: "Штраф 0,1% в день [1].")
    return TestClient(api.app)  # без with: lifespan не запускается, используем подготовленный rag


def test_ask_strict_and_debug(client):
    r = client.post("/ask", json={"question": "штраф за просрочку доставки"})
    assert r.status_code == 200 and set(r.json()) == {"answer", "sources", "confidence"}
    d = client.post("/ask?debug=true", json={"question": "штраф за просрочку доставки"}).json()
    assert "timings_ms" in d and "route" in d


def test_ask_rejects_empty(client):
    assert client.post("/ask", json={"question": "  "}).status_code == 400


def test_update_indexes_new_file(client):
    r = client.post("/update", data={"company": "logistics"},
                    files={"file": ("new.txt", "Регламент склада: приёмка с 9 до 18.".encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["chunks"] == 1
    assert client.post("/update", data={"company": "nope"},
                       files={"file": ("x.txt", b"x", "text/plain")}).status_code == 400
