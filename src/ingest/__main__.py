"""Индексация папки с документами: python -m src.ingest [путь_к_docs] [--force]

Повторный запуск пропускает уже проиндексированные файлы (--force переиндексирует всё).
"""
import sys
import time
from pathlib import Path

from .. import config
from ..retrieve.index import Index


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else config.DOCS_DIR
    t0 = time.perf_counter()
    n = Index().ingest_dir(root, force="--force" in sys.argv)
    print(f"Готово: {n} новых чанков за {time.perf_counter() - t0:.1f} с")


if __name__ == "__main__":
    main()
