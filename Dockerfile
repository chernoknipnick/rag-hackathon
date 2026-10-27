FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-сборка torch: в разы меньше CUDA-версии, ноутбук жюри без GPU
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/hf_cache
COPY src ./src
COPY scripts ./scripts
COPY eval ./eval

# Модели эмбеддингов и reranker скачиваются при сборке образа,
# чтобы при первом запуске у жюри ничего не качалось.
ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python scripts/download_models.py; fi

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=20 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
