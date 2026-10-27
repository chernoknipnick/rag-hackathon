"""Нарезка текста на чанки и стабильные id."""
import re
import uuid

from .. import config


def make_id(doc: str, page, n: int) -> str:
    """Один и тот же фрагмент всегда получает один и тот же id (важно для переиндексации)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc}|{page}|{n}"))


def normalize(text: str) -> str:
    text = text.replace(" ", " ").replace("­", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Режет по ~size символов, стараясь попасть на границу строки или предложения."""
    size = size or config.CHUNK_CHARS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    text = normalize(text)
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]
