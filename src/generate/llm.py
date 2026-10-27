"""Локальная LLM через Ollama (https://ollama.com). Внешних API в рантайме нет.

Интерфейс генератора: callable(question, hits) -> str. В тестах подменяется лямбдой.
"""
import requests

from .. import config
from ..schema import Hit
from .prompt import build_messages


def _post(messages: list[dict], max_tokens: int) -> str:
    payload = {
        "model": config.LLM_MODEL,
        "stream": False,
        "keep_alive": "30m",  # не выгружать модель между запросами
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": config.LLM_NUM_CTX},
        "messages": messages,
    }
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def generate(question: str, hits: list[Hit]) -> str:
    return _post(build_messages(question, hits), config.LLM_MAX_TOKENS)


def warmup() -> bool:
    """Загрузить модель в память заранее, чтобы первый вопрос жюри не ждал 10–30 секунд."""
    try:
        _post([{"role": "user", "content": "Ответь одним словом: готов?"}], 3)
        return True
    except Exception:
        return False
