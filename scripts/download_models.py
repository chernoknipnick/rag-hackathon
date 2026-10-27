"""Скачать модели эмбеддингов и reranker в кэш Hugging Face заранее.

Запуск: python scripts/download_models.py
В Docker вызывается при сборке образа (PRELOAD_MODELS=1), чтобы жюри не ждало загрузку при старте.
LLM скачивается отдельно: ollama pull $LLM_MODEL.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402


def main():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    print(f"[models] эмбеддер: {config.EMBED_MODEL}", flush=True)
    SentenceTransformer(config.EMBED_MODEL, device="cpu")
    if config.USE_RERANKER:
        print(f"[models] reranker: {config.RERANK_MODEL}", flush=True)
        CrossEncoder(config.RERANK_MODEL, device="cpu")
    print("[models] готово", flush=True)


if __name__ == "__main__":
    main()
