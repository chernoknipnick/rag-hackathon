"""Роль 1: парсеры и метаданные."""
import pytest

from src.ingest.chunking import split_text
from src.ingest.parsers import parse_file


def test_docx_sections_paragraphs_and_tables_in_place(docs):
    chunks = parse_file(docs / "logistics" / "dogovor_transgruz.docx", docs)
    assert all(c.company == "logistics" and c.doc == "logistics/dogovor_transgruz.docx" for c in chunks)
    fine = next(c for c in chunks if "0,1%" in c.text)
    assert fine.section == "5. Ответственность сторон"
    assert isinstance(fine.page, int) and fine.location.startswith("абзац")
    row = next(c for c in chunks if c.kind == "table_row")
    assert "Нарушение: Повреждение груза" in row.text and row.section == "5. Ответственность сторон"
    order = [c.text[:12] for c in chunks]
    assert order.index(row.text[:12]) < order.index(next(c for c in chunks if "арбитраж" in c.text).text[:12])


def test_xlsx_finds_real_header_under_title_row(docs):
    chunks = parse_file(docs / "logistics" / "tarify.xlsx", docs)
    kazan = next(c for c in chunks if "Казань" in c.text)
    assert "Маршрут: Москва — Казань" in kazan.text and "Цена, руб: 12 400" in kazan.text
    assert kazan.kind == "table_row" and kazan.section == "Тарифы"
    assert kazan.page == 3 and kazan.location == "Тарифы, строка 3"  # как в Excel: заголовок, шапка, Казань


def test_csv_rows_keep_headers(docs):
    chunks = parse_file(docs / "retail" / "stock.csv", docs)
    fc = next(c for c in chunks if "Артикул: FC-1" in c.text)
    assert "Остаток: 14" in fc.text and fc.page == 2


def test_json_finds_nested_records_and_flattens(docs):
    chunks = parse_file(docs / "retail" / "cards.json", docs)
    assert len(chunks) == 2
    assert "specs.weight_g: 30" in chunks[0].text and chunks[0].page == 1


def test_markdown_sections(docs):
    chunks = parse_file(docs / "retail" / "return_policy.md", docs)
    assert {c.section for c in chunks} >= {"Политика возврата", "Возврат после 14 дней"}
    assert all(isinstance(c.page, int) for c in chunks)


def test_pdf_page_numbers(docs):
    pytest.importorskip("fitz")
    chunks = parse_file(docs / "tehnoprom" / "stanok_tp400.pdf", docs)
    assert {c.page for c in chunks} == {1, 2}
    assert next(c for c in chunks if "1200" in c.text).location == "стр. 2"


def test_ids_are_stable(docs):
    a = [c.id for c in parse_file(docs / "retail" / "stock.csv", docs)]
    b = [c.id for c in parse_file(docs / "retail" / "stock.csv", docs)]
    assert a == b and len(set(a)) == len(a)


def test_split_text_respects_size():
    parts = split_text("Предложение номер один. " * 200, size=500, overlap=50)
    assert len(parts) > 1 and all(len(p) <= 500 for p in parts)
