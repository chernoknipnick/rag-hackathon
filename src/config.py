"""Все настройки берутся из переменных окружения (см. .env.example).

Владелец: роль 5 (сборка). Значения под себя меняйте в .env, а не здесь:
так меньше конфликтов в Git.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path):
    """Читает .env (KEY=VALUE, комментарии после « #»). Переменные окружения важнее файла."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


# --- пути ---
DOCS_DIR = Path(_env("DOCS_DIR", str(ROOT / "docs")))          # docs/<компания>/файлы
INDEX_DIR = Path(_env("INDEX_DIR", str(ROOT / "index_data")))   # chunks.jsonl + vectors.npy

# --- модели ---
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-m3")
QUERY_PREFIX = _env("QUERY_PREFIX", "")      # для e5: "query: "
PASSAGE_PREFIX = _env("PASSAGE_PREFIX", "")  # для e5: "passage: "
RERANK_MODEL = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
USE_RERANKER = _env("USE_RERANKER", "1") == "1"
RERANK_MAX_LENGTH = _int("RERANK_MAX_LENGTH", 512)

# --- LLM (Ollama, локально) ---
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:3b-instruct")
LLM_TIMEOUT = _int("LLM_TIMEOUT", 60)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 150)   # короткий ответ = быстрый ответ на CPU
LLM_NUM_CTX = _int("LLM_NUM_CTX", 4096)

# --- поиск ---
TOP_K_RETRIEVE = _int("TOP_K_RETRIEVE", 12)    # кандидатов в реранкер (меньше = быстрее)
TOP_K_CONTEXT = _int("TOP_K_CONTEXT", 3)       # чанков в промпт LLM (меньше = быстрее prefill)
RRF_K = 60
CHUNK_CHARS = _int("CHUNK_CHARS", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)

# --- уверенность ---
MIN_CONFIDENCE = _float("MIN_CONFIDENCE", 0.3)       # ниже -> «Нет в документах» без вызова LLM
UNSURE_CONFIDENCE = _float("UNSURE_CONFIDENCE", 0.5)  # ниже -> пометка «не уверен» в debug и UI

# --- роутер ---
ROUTER_NCM_MARGIN = _float("ROUTER_NCM_MARGIN", 0.03)  # разрыв 1-го и 2-го центроида; меньше -> все компании
ROUTER_CENTROIDS = _int("ROUTER_CENTROIDS", 3)         # центроидов на компанию (k-means)
ROUTER_FALLBACK_ALL = _env("ROUTER_FALLBACK_ALL", "1") == "1"  # низкий скор в компании -> повторный поиск по всем

# --- прочее ---
CACHE_SIZE = _int("CACHE_SIZE", 256)
WARMUP = _env("WARMUP", "1") == "1"   # прогрев моделей при старте API

NO_ANSWER = "Нет в документах"

# Папка в docs/ = компания. Отображаемые имена идут в шапку чанка и в ответ.
# В ТЗ два набора названий (легенда и примеры вопросов), поэтому указаны оба.
COMPANY_TITLES: dict[str, str] = {
    "tehnoprom": "ТехноПром / Миэлта (производство оборудования)",
    "logistics": "ЛогистикСервис / СмартАгро (логистика, договоры, тарифы)",
    "retail": "РозницаОнлайн / Агродинамика (розница, товары, возвраты)",
}
COMPANIES = list(COMPANY_TITLES)
