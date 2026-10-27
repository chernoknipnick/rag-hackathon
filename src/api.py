"""FastAPI: POST /ask, POST /update (бонус B), GET /health, GET / (веб-интерфейс для демо).

Владелец: роль 5 (сборка).

POST /ask отвечает строго в формате ТЗ: {"answer", "sources": [{"doc", "page", "score"}], "confidence"}.
POST /ask?debug=true добавляет компанию, способ роутинга, время по шагам, «не уверен».
"""
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config
from .ingest.parsers import PARSERS
from .pipeline import RAG
from .schema import strict_response

MAX_UPLOAD = 50 * 1024 * 1024
_lock = threading.Lock()  # индекс и кэш меняются через /update -> один запрос за раз
rag: RAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAG()
    if not rag.index.chunks and config.DOCS_DIR.exists():
        print("[startup] индекс пуст, индексирую docs/ ...", flush=True)
        rag.index.ingest_dir(config.DOCS_DIR)
        rag.on_index_changed()
    if config.WARMUP:
        rag.warmup()
    print(f"[startup] готово: {len(rag.index.chunks)} чанков", flush=True)
    yield


app = FastAPI(title="Корпоративный RAG", lifespan=lifespan)


class Question(BaseModel):
    question: str


class Source(BaseModel):
    doc: str
    page: int | None
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(rag.index.chunks) if rag else 0}


@app.post("/ask", response_model=None)
def ask(q: Question, debug: bool = False):
    if not q.question.strip():
        raise HTTPException(400, "Пустой вопрос")
    with _lock:
        result = rag.ask(q.question)
    return result if debug else Answer(**strict_response(result))


@app.post("/update")
def update(company: str = Form(...), file: UploadFile = File(...)):
    """Добавить документ и переиндексировать только его (инкрементально)."""
    if company not in config.COMPANY_TITLES:
        raise HTTPException(400, f"company должна быть одной из {list(config.COMPANY_TITLES)}")
    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in PARSERS:
        raise HTTPException(400, f"Формат не поддерживается: {sorted(PARSERS)}")
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Файл больше 50 МБ")

    dest = config.DOCS_DIR / company / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    t0 = time.perf_counter()
    with _lock:
        n = rag.index.ingest_file(dest, config.DOCS_DIR)
        rag.on_index_changed()
    return {"doc": f"{company}/{name}", "chunks": n, "seconds": round(time.perf_counter() - t0, 2)}


PAGE = (Path(__file__).parent / "web.html")


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.read_text(encoding="utf-8")
