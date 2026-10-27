"""Роль 1 — данные: парсинг файлов, чанки, метаданные."""
from .parsers import PARSERS, parse_file, rel_path

__all__ = ["PARSERS", "parse_file", "rel_path"]
