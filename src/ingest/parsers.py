"""Парсеры файлов -> list[Chunk] с метаданными.

Владелец: роль 1 (данные).

Каждый парсер отдаёт блоки Block(text, page, location, section, kind):
  PDF   — страница (page = номер страницы, OCR для сканов);
  DOCX  — абзацы под одним заголовком (page = номер первого абзаца), таблицы на своём месте;
  XLSX/CSV — одна строка = один чанк с заголовками колонок (page = номер строки);
  JSON  — одна запись = один чанк (page = номер записи), ищет список записей на любой глубине;
  MD/TXT — разделы по заголовкам Markdown, иначе фрагменты;
  PNG/JPG — OCR.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .. import config
from ..schema import Chunk
from .chunking import make_id, normalize, split_text


@dataclass
class Block:
    text: str
    page: int | None
    location: str
    section: str = ""
    kind: str = "text"


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


# ---------- OCR (нужен pytesseract + бинарник tesseract с rus) ----------
def _ocr_image(img) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img, lang="rus+eng")
    except Exception:
        return ""


def _ocr_pdf_page(page) -> str:
    try:
        import io

        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        return _ocr_image(Image.open(io.BytesIO(pix.tobytes("png"))))
    except Exception:
        return ""


# ---------- PDF ----------
def _pdf(path: Path) -> Iterator[Block]:
    import fitz  # PyMuPDF
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf, 1):
            text = page.get_text("text")
            kind = "text"
            if len(text.strip()) < 30:  # нет текстового слоя -> скан -> OCR
                ocr = _ocr_pdf_page(page)
                if ocr.strip():
                    text, kind = ocr, "image"
            if text.strip():
                yield Block(text, i, f"стр. {i}", kind=kind)


# ---------- DOCX ----------
_CLAUSE = re.compile(r"^(\d+(\.\d+)*\.?)\s+\S")  # «5.», «5.2», «7.1.3 Ответственность сторон»


def _is_heading(par) -> bool:
    try:
        style = (par.style.name or "").lower() if par.style is not None else ""
    except Exception:
        style = ""
    if style.startswith(("heading", "заголовок", "title")):
        return True
    t = par.text.strip()
    if not t or len(t) > 120:
        return False
    if t.isupper() and len(t) > 3:
        return True
    return bool(_CLAUSE.match(t)) and len(t) < 80 and not t.endswith((".", ";", ","))


def _table_rows(rows: list[list[str]], section: str, label: str, page: int | None) -> Iterator[Block]:
    """Строка таблицы DOCX -> чанк «шапка: значение». page = номер абзаца перед таблицей."""
    def dedupe(r):  # объединённые ячейки python-docx отдаёт повторами одного текста
        return [c.strip() if i == 0 or c.strip() != r[i - 1].strip() else "" for i, c in enumerate(r)]

    rows = [dedupe(r) for r in rows if any(c.strip() for c in r)]
    if not rows:
        return
    head, body = rows[0], rows[1:] or [rows[0]]
    for n, r in enumerate(body, 2):
        parts = [f"{h}: {v}" if h and h != v else v for h, v in zip(head, r) if v]
        if parts:
            yield Block("; ".join(parts), page, f"{label}, строка {n}", section, "table_row")


def _docx(path: Path) -> Iterator[Block]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    section, buf, size, first, last, par_no, table_no = "", [], 0, 1, 1, 0, 0

    def flush():
        nonlocal buf, size
        if buf:
            loc = f"абзац {first}" if first == last else f"абзацы {first}–{last}"
            yield Block("\n".join(buf), first, loc, section)
        buf, size = [], 0

    for el in d.element.body.iterchildren():  # абзацы и таблицы в порядке документа
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            par = Paragraph(el, d)
            t = par.text.strip()
            if not t:
                continue
            par_no += 1
            if _is_heading(par):
                yield from flush()
                section = t[:100]
                continue
            if not buf:
                first = par_no
            last = par_no
            buf.append(t)
            size += len(t)
            if size >= config.CHUNK_CHARS:
                yield from flush()
        elif tag == "tbl":
            yield from flush()
            table_no += 1
            rows = [[c.text for c in r.cells] for r in Table(el, d).rows]
            yield from _table_rows(rows, section, f"таблица {table_no}", max(par_no, 1))
    yield from flush()


# ---------- XLSX / CSV ----------
def _fix_header(df):
    """Если над таблицей есть строки-заголовки («Тарифная сетка 2024»), ищем настоящую шапку."""
    cols = [str(c) for c in df.columns]
    if sum(c.startswith("Unnamed") for c in cols) <= len(cols) // 2:
        return df
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        if row.notna().sum() >= max(2, len(cols) // 2):
            new = df.iloc[i + 1:].copy()
            new.columns = [str(v) if str(v) != "nan" else f"col{j}" for j, v in enumerate(row)]
            return new
    return df


def _frame_rows(df, sheet: str = "") -> Iterator[Block]:
    """Номер строки = номер строки в Excel/CSV (pandas-индекс 0 — это строка 2 файла)."""
    df = _fix_header(df.reset_index(drop=True))
    cols = [str(c) for c in df.columns]
    for label, *row in df.itertuples(index=True):
        n = int(label) + 2
        parts = [f"{c}: {v}" for c, v in zip(cols, row) if str(v).strip() and str(v) != "nan"]
        if parts:
            loc = f"{sheet}, строка {n}" if sheet else f"строка {n}"
            yield Block("; ".join(parts), n, loc, sheet, "table_row")


def _xlsx(path: Path) -> Iterator[Block]:
    import pandas as pd
    for sheet, df in pd.read_excel(path, sheet_name=None, dtype=str).items():
        yield from _frame_rows(df, str(sheet))


def _csv(path: Path) -> Iterator[Block]:
    import pandas as pd
    for enc in ("utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return
    yield from _frame_rows(df)


# ---------- JSON ----------
def _flatten(obj, prefix: str = "") -> list[str]:
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out += _flatten(v, f"{prefix}{k}.")
        return out
    if isinstance(obj, list) and not any(isinstance(x, (dict, list)) for x in obj):
        return [f"{prefix[:-1]}: {', '.join(map(str, obj))}"] if obj else []
    if isinstance(obj, list):
        return [f"{prefix[:-1]}: {json.dumps(obj, ensure_ascii=False)}"]
    return [f"{prefix[:-1]}: {obj}"] if str(obj).strip() else []


def _records(data):
    """Самый длинный список словарей на любой глубине = записи (товары, заказы)."""
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        return data
    best = None
    if isinstance(data, dict):
        for v in data.values():
            r = _records(v)
            if r and (best is None or len(r) > len(best)):
                best = r
    return best


def _json(path: Path) -> Iterator[Block]:
    data = json.loads(_read_text(path))
    records = _records(data) or [data]
    for n, item in enumerate(records, 1):
        text = "; ".join(_flatten(item)) if isinstance(item, (dict, list)) else str(item)
        if text.strip():
            yield Block(text, n, f"запись {n}", kind="record")


# ---------- TXT / MD ----------
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.M)


def _txt(path: Path) -> Iterator[Block]:
    text = _read_text(path)
    parts = _MD_HEADING.split(text) if path.suffix.lower() == ".md" else [text]
    sections = [("", parts[0])] + list(zip(parts[1::2], parts[2::2]))
    n = 0
    for title, body in sections:
        for piece in split_text(body):
            n += 1
            loc = f"раздел «{title.strip()}»" if title else f"фрагмент {n}"
            yield Block(piece, n, loc, title.strip())


# ---------- изображения ----------
def _image(path: Path) -> Iterator[Block]:
    from PIL import Image
    text = _ocr_image(Image.open(path))
    if text.strip():
        yield Block(text, 1, "изображение (OCR)", kind="image")


PARSERS = {
    ".pdf": _pdf, ".docx": _docx, ".xlsx": _xlsx, ".csv": _csv,
    ".json": _json, ".txt": _txt, ".md": _txt,
    ".png": _image, ".jpg": _image, ".jpeg": _image,
}


def rel_path(path: Path, root: Path | None = None) -> str:
    root = Path(root or config.DOCS_DIR)
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def parse_file(path, root: Path | None = None) -> list[Chunk]:
    path = Path(path)
    fn = PARSERS.get(path.suffix.lower())
    if fn is None:
        return []
    rel = rel_path(path, root)
    parts = rel.split("/")
    company = parts[0] if len(parts) > 1 else "unknown"
    chunks = []
    for b in fn(path):
        pieces = [normalize(b.text)] if b.kind in ("table_row", "record") else split_text(b.text)
        for n, piece in enumerate(pieces):
            if piece:
                chunks.append(Chunk(make_id(rel, b.location, n), piece, rel, company,
                                    b.page, b.location, b.section, b.kind))
    return chunks
