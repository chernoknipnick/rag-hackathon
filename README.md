# Корпоративный RAG: единый помощник для группы компаний

Чат-бот по документам трёх компаний. Определяет компанию по вопросу, ищет в её документах
и отвечает с источником (файл, страница) и уверенностью 0–1. Если данных нет, честно отвечает
«Нет в документах». Работает полностью локально: open-source модели, без внешних API в рантайме.

Соответствие пунктам ТЗ и статус по каждому: [TZ.md](TZ.md). Как работаем в Git: [CONTRIBUTING.md](CONTRIBUTING.md).

## Архитектура

```
Индексация (один раз, заранее)
  docs/<компания>/*  ->  парсинг (PDF + OCR, DOCX, XLSX/CSV, JSON, MD/TXT, изображения)
                     ->  чанки с метаданными (компания, файл, страница/абзац/строка, раздел)
                     ->  эмбеддинги bge-m3 (numpy) + BM25  ->  index_data/

Ответ на вопрос (каждый запрос)
  вопрос -> кэш -> эмбеддинг вопроса
         -> роутер компании: правила + NCM (центроиды документов компаний)
         -> гибридный поиск: векторный + BM25, слияние RRF (top-12)
         -> reranker bge-reranker-base -> лучший скор = confidence
         -> скор < порога ? «Нет в документах» без вызова LLM
         -> ничего уверенного в выбранной компании ? повторный поиск по всем компаниям
         -> локальная LLM (Qwen2.5 через Ollama), ответ со ссылками [n]
         -> источники = процитированные фрагменты; самопроверка чисел против документов
         -> {"answer", "sources": [{"doc", "page", "score"}], "confidence"}
```

## Структура и владельцы

| Папка | Что внутри | Владелец |
|---|---|---|
| `src/schema.py` | контракты: `Chunk`, `Hit`, `Route`, формат ответа | вся команда, по договорённости |
| `src/ingest/` | парсеры форматов, чанкинг, метаданные | роль 1 — данные |
| `src/retrieve/` | эмбеддер, векторное хранилище, BM25, гибридный поиск, reranker | роль 2 — поиск |
| `src/generate/` | промпт, LLM, порог, confidence, источники, самопроверка | роль 3 — генерация |
| `src/router/` | правила + NCM | роль 4 — роутер и оценка |
| `eval/` | метрики: hit@k, MRR, точность роутера, отказы, время | роль 4 |
| `src/pipeline.py`, `src/api.py`, `src/web.html`, Docker, README | склейка, API, демо | роль 5 — сборка |

## Быстрый старт (Mac / Linux, разработка)

```bash
brew install ollama tesseract tesseract-lang        # Linux: apt install tesseract-ocr tesseract-ocr-rus; ollama: ollama.com
ollama pull qwen2.5:3b-instruct
python3 -m venv .venv && source .venv/bin/activate   # Python 3.10–3.12
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # на Mac: pip install torch==2.5.1
pip install -r requirements-dev.txt
cp .env.example .env
python -m src.ingest docs              # индексация (повторный запуск продолжит с места остановки)
uvicorn src.api:app --port 8000        # веб-интерфейс: http://localhost:8000
pytest -q                              # тесты на фейковых моделях, без скачивания
```

## Запуск в Docker (для жюри, CPU)

```bash
docker compose up --build
```

Модели эмбеддингов и reranker скачиваются при сборке образа, LLM — при первом старте контейнера `ollama`.
API стартует, когда модель готова, и сам индексирует `docs/`, если индекс пуст.

Чтобы у жюри старт был мгновенным, положите заранее собранный индекс в `index_data/`
(собирается командой `python -m src.ingest docs` на тех же документах и той же модели эмбеддингов).

## API

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "какой вес станка ТП-400?"}'
# {"answer": "1200 кг", "sources": [{"doc": "tehnoprom/stanok_tp400.pdf", "page": 12, "score": 0.92}], "confidence": 0.92}

curl -X POST 'localhost:8000/ask?debug=true' -H 'Content-Type: application/json' -d '{"question": "..."}'
# + company, route, timings_ms (время по шагам), unsure, location источника

curl -X POST localhost:8000/update -F company=logistics -F file=@new_contract.docx
# {"doc": "logistics/new_contract.docx", "chunks": 14, "seconds": 3.2}
```

`page` — номер страницы для PDF; для остальных форматов номер абзаца (DOCX), строки (XLSX/CSV)
или записи (JSON). Человекочитаемая подпись («абзацы 3–9», «Тарифы, строка 7») — в `location` (режим debug).

## Оценка

```bash
python -m eval.evaluate eval/questions.json           # hit@1, hit@3, MRR, точность роутера (быстро, без LLM)
python -m eval.evaluate eval/questions.json --full    # + доля верных отказов, факты в ответе, время p50/p95
```

Формат вопросов — в `eval/questions.example.json`, генерация синтетических вопросов — `eval/SYNTHETIC_PROMPT.md`.

Что подбирать: `MIN_CONFIDENCE` (баланс «не знаю» и ответов), `TOP_K_RETRIEVE` и `TOP_K_CONTEXT` (скорость),
`CHUNK_CHARS`, ключевые слова роутера (`src/router/rules.py`), `ROUTER_NCM_MARGIN`, модель LLM.

## Модели и лицензии

| Компонент | Модель | Лицензия |
|---|---|---|
| Эмбеддинги | BAAI/bge-m3 | MIT |
| Reranker | BAAI/bge-reranker-base | MIT |
| LLM | Qwen2.5-3B-Instruct (Ollama) | Qwen Research License (некоммерческая) |
| LLM, альтернатива | Qwen2.5-1.5B / 7B-Instruct | Apache 2.0 |

Перед сдачей проверьте лицензию выбранной модели на её странице.

## Честно о подготовке

В рантайме внешние API не используются. На этапе подготовки код писали с помощью Claude,
синтетические вопросы для самопроверки генерировали им же (как разрешает ТЗ). Каркас пайплайна
подготовлен до хакатона (тег `pre-hackathon`), работа с данными хакатона — на месте.

## Известные ограничения

- Таблица на картинке распознаётся OCR как текст, структура строк и столбцов не восстанавливается.
- Расчёт по тарифной сетке с диапазонами (например, «до 1000 кг») пока не делается: строка тарифа находится поиском, LLM отвечает по ней.
- Голосовой ввод в веб-интерфейсе использует распознавание речи браузера (в Chrome оно идёт через сервер Google). Для полностью офлайн-режима нужен локальный Whisper.
