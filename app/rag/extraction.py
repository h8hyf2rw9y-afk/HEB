"""PDF text extraction with PyMuPDF, preserving page numbers, sections and tables.

Pipeline per document:
1. Read text lines (with font size / bold / bbox) and detect tables with
   ``page.find_tables()``. Lines inside a table area are not read as prose.
2. Drop running headers/footers: lines repeated on most pages (digits ignored)
   and "Página N" markers.
3. Headings are bold or larger-than-body lines; consecutive heading lines of the
   same PyMuPDF block are merged (multi-line questions such as "P18. ...").
4. Each table row becomes its own block ("Header: value | Header: value."), so a
   row such as "Electrodomésticos ... | Plazo: 30 días naturales" stays intact
   and is cited on its own. Two-column key/value tables (document cover) and
   change-log tables are tagged so they are not used for conflict detection.
5. Document facts (code, version, effective date, cited document codes,
   supersession and precedence sentences) are taken literally from the text.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from .models import DocumentInfo
from .text import normalize, split_sentences

try:  # PyMuPDF >= 1.24 exposes ``pymupdf``; older releases only ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - depends on installed version
    import fitz as pymupdf  # type: ignore[no-redef]


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


BlockKind = Literal["text", "table_row", "metadata_row", "changelog_row"]

_NUMBERED_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?)|(?:[IVXLC]+\.)|(?:P\d+\.)|(?:secci[oó]n|cap[ií]tulo|art[ií]culo|anexo)\b)\s*\S",
    re.IGNORECASE,
)
_PAGE_MARKER = re.compile(r"^p[aá]gina\s+\d+(\s+de\s+\d+)?$", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^(?:[•▪◦·\-–]|\d{1,2}[.)]?)$")
_DOC_CODE = re.compile(r"\b[A-Z]{2,4}-[A-Z]{2,4}-\d{3}\b")
_BOLD_FLAG = 1 << 4


@dataclass
class TextLine:
    text: str
    size: float
    bold: bool
    y: float = 0.0
    x: float = 0.0
    block_id: int = 0


@dataclass
class PageBlock:
    """A paragraph-like block (or one table row) tagged with its active section."""

    page: int
    section: str | None
    text: str
    kind: BlockKind = "text"
    label: str | None = None  # first cell of a table row


@dataclass
class ExtractedDocument:
    document: str
    path: Path | None
    blocks: list[PageBlock] = field(default_factory=list)
    info: DocumentInfo | None = None

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


# ----------------------------------------------------------------- helpers
def _clean_cell(raw: str) -> str:
    """Join the lines of a table cell, repairing narrow-column word breaks
    ("Versi\\nón" -> "Versión", "Priori\\ndad" -> "Prioridad")."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out = ""
    for ln in lines:
        if out and len(ln) <= 3 and ln.isalpha() and ln.islower() and out[-1:].isalpha():
            out += ln
        else:
            out = f"{out} {ln}" if out else ln
    return re.sub(r"\s+", " ", out).strip()


def _is_heading(line: TextLine, body_size: float) -> bool:
    text = line.text.strip()
    words = text.split()
    if not words or len(text) < 3:
        return False
    numbered = bool(_NUMBERED_HEADING.match(text))
    larger = line.size >= body_size * 1.15
    max_words = 30 if (line.bold and numbered) else 14
    if len(words) > max_words:
        return False
    if text.endswith((".", ",", ";")) and not numbered:
        return False
    return larger or (line.bold and (numbered or text[:1].isupper() or text[:1] in "¿¡"))


def _span_bold(span: dict) -> bool:
    return bool((span.get("flags", 0) & _BOLD_FLAG) or "bold" in span.get("font", "").lower())


def _inside(rect: tuple[float, float, float, float], x: float, y: float) -> bool:
    x0, y0, x1, y1 = rect
    return x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1


def _read_lines(page: "pymupdf.Page", skip_rects: list[tuple]) -> list[TextLine]:
    lines: list[TextLine] = []
    for block_id, block in enumerate(page.get_text("dict").get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            x0, y0, x1, y1 = line["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(_inside(r, cx, cy) for r in skip_rects):
                continue
            lines.append(
                TextLine(
                    text=" ".join(s["text"].strip() for s in spans),
                    size=max(s.get("size", 0.0) for s in spans),
                    bold=all(_span_bold(s) for s in spans),
                    y=y0,
                    x=x0,
                    block_id=block_id,
                )
            )
    return lines


@dataclass
class _Table:
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]
    bold: list[list[bool]]

    @property
    def y(self) -> float:
        return self.bbox[1]


def _read_tables(page: "pymupdf.Page") -> list[_Table]:
    try:
        # find_tables relies on ruling lines; skip the (slow) detector on pages
        # without vector drawings.
        if not page.get_drawings():
            return []
        found = page.find_tables().tables
    except Exception:  # table detection is best-effort; prose extraction still works
        return []
    tables: list[_Table] = []
    for t in found:
        rows, bold = [], []
        for row in t.rows:
            texts, flags = [], []
            for cell in row.cells:
                if cell is None:
                    texts.append("")
                    flags.append(False)
                    continue
                rect = pymupdf.Rect(cell)
                texts.append(_clean_cell(page.get_text("text", clip=rect)))
                spans = [
                    s
                    for b in page.get_text("dict", clip=rect).get("blocks", [])
                    for ln in b.get("lines", [])
                    for s in ln.get("spans", [])
                    if s.get("text", "").strip()
                ]
                flags.append(bool(spans) and all(_span_bold(s) for s in spans))
            if any(texts):
                rows.append(texts)
                bold.append(flags)
        if rows:
            tables.append(_Table(bbox=tuple(t.bbox), rows=rows, bold=bold))
    return tables


def _table_blocks(table: _Table, page_no: int, section: str | None) -> list[PageBlock]:
    # Drop spacer columns that are empty in every row.
    width = max(len(r) for r in table.rows)
    keep = [c for c in range(width) if any(c < len(r) and r[c] for r in table.rows)]
    rows = [[r[c] if c < len(r) else "" for c in keep] for r in table.rows]
    bold = [[b[c] if c < len(b) else False for c in keep] for b in table.bold]
    ncol = len(keep)
    first_col_bold = all(b[0] for b in bold if b)
    other_bold = any(any(b[1:]) for b in bold)
    if ncol == 2 and first_col_bold and not other_bold:
        # Key/value table (e.g. document cover: "Fecha de vigencia | 12 de enero de 2026").
        pairs = [(r[0], r[1] if len(r) > 1 else "") for r in rows]
        return [
            PageBlock(page_no, section, f"{k}: {v.rstrip('.')}.", kind="metadata_row", label=k)
            for k, v in pairs
            if k
        ]
    header, body = rows[0], rows[1:]
    header_norm = {normalize(h) for h in header}
    kind: BlockKind = "changelog_row" if {"version", "fecha"} <= header_norm else "table_row"
    blocks = []
    for r in body:
        cells = [f"{h}: {v}" if h else v for h, v in zip(header, r) if v]
        if cells:
            cells[-1] = cells[-1].rstrip(".")
        if cells:
            blocks.append(PageBlock(page_no, section, " | ".join(cells) + ".", kind=kind, label=r[0] or None))
    return blocks


def _repeated_lines(pages: list[list[TextLine]]) -> set[str]:
    """Running headers/footers: lines present on >= 60% of pages (docs with >= 3 pages)."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({re.sub(r"\d+", "#", ln.text.strip()) for ln in lines})
    return {t for t, n in counts.items() if n >= 0.6 * len(pages)}


# --------------------------------------------------------------- extraction
def extract_pdf(path: str | Path) -> ExtractedDocument:
    """Extract text blocks with (page, section) metadata from one PDF."""
    path = Path(path)
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises several exception types
        raise PdfExtractionError(f"No se pudo abrir el PDF {path.name}: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise PdfExtractionError(f"El PDF {path.name} está protegido con contraseña")
        page_tables = [_read_tables(page) for page in doc]
        page_lines = [
            _read_lines(page, [t.bbox for t in tables]) for page, tables in zip(doc, page_tables)
        ]
        page_count = doc.page_count

    repeated = _repeated_lines(page_lines)
    sizes = [ln.size for lines in page_lines for ln in lines for _ in range(max(1, len(ln.text)))]
    body_size = statistics.median(sizes) if sizes else 11.0

    result = ExtractedDocument(document=path.name, path=path)
    empty_pages: list[int] = []
    section: str | None = None
    title: str | None = None
    for page_no, (lines, tables) in enumerate(zip(page_lines, page_tables), start=1):
        lines = [
            ln for ln in lines
            if re.sub(r"\d+", "#", ln.text.strip()) not in repeated and not _PAGE_MARKER.match(ln.text.strip())
        ]
        if not lines and not tables:
            empty_pages.append(page_no)
            continue
        # Keep PyMuPDF's reading order for prose; place each table before the first
        # line that starts below it. Bare list markers ("•", "1") are dropped.
        lines = [ln for ln in lines if not _LIST_MARKER.match(ln.text.strip())]
        items: list[tuple[float, float, object]] = [(ln.y, ln.x, ln) for ln in lines]
        for t in sorted(tables, key=lambda t: t.y):
            pos = next((k for k, it in enumerate(items) if isinstance(it[2], TextLine) and it[0] > t.y), len(items))
            items.insert(pos, (t.y, 0.0, t))

        buffer: list[str] = []
        heading_parts: list[str] = []
        last_block: int | None = None
        last_was_heading = False

        def flush_text() -> None:
            if buffer:
                result.blocks.append(PageBlock(page_no, section, " ".join(buffer)))
                buffer.clear()

        for _, _, item in items:
            if isinstance(item, _Table):
                flush_text()
                result.blocks.extend(_table_blocks(item, page_no, section))
                last_was_heading = False
                last_block = None
                continue
            line: TextLine = item  # type: ignore[assignment]
            text = line.text.strip()
            if _is_heading(line, body_size) or (
                last_was_heading and line.bold and line.block_id == last_block
            ):
                flush_text()
                if last_was_heading and line.block_id == last_block:
                    heading_parts.append(text)  # continuation of a multi-line heading
                else:
                    heading_parts = [text]
                section = " ".join(heading_parts)
                if title is None and line.size >= body_size * 1.3:
                    title = text
                last_was_heading = True
            else:
                if last_block is not None and line.block_id != last_block:
                    flush_text()
                buffer.append(text)
                last_was_heading = False
            last_block = line.block_id
        flush_text()

    result.info = build_document_info(result, page_count=page_count, title=title, empty_pages=empty_pages)
    return result


# ---------------------------------------------------------------------------
# Document facts used for conflict precedence. Only phrases literally present in
# the document are recorded; nothing is inferred from file metadata.
# ---------------------------------------------------------------------------
_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
_DATE = r"(?:(\d{1,2})\s+de\s+([a-záéíóú]+)\s+(?:de|del)\s+(\d{4})|(\d{4})-(\d{2})-(\d{2})|(\d{1,2})/(\d{1,2})/(\d{4}))"
_EFFECTIVE_RE = re.compile(
    r"(?:vigente|vigencia|entra en vigor|en vigor|efectiv[oa])[^.\n]{0,40}?" + _DATE, re.IGNORECASE
)
_SUPERSEDES_RE = re.compile(
    r"\b(?:reemplaza|sustituye|deja sin efecto|deroga|actualiza y reemplaza)\b", re.IGNORECASE
)
_PRECEDENCE_RE = re.compile(
    r"\ben caso de (?:duda o (?:de )?)?(?:conflicto|discrepancia|contradicci[oó]n|diferencia)\b"
    r"|\bprevalecer?[aá]?n?\b",
    re.IGNORECASE,
)


def _parse_date(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    g = match.groups()
    try:
        if g[0]:
            month = _MONTHS.get(g[1].lower().replace("é", "e"))
            if not month:
                return None
            return date(int(g[2]), month, int(g[0])).isoformat()
        if g[3]:
            return date(int(g[3]), int(g[4]), int(g[5])).isoformat()
        if g[6]:
            return date(int(g[8]), int(g[7]), int(g[6])).isoformat()  # dd/mm/yyyy
    except ValueError:
        return None
    return None


def _parse_effective_date(text: str) -> str | None:
    return _parse_date(text, _EFFECTIVE_RE)


def build_document_info(
    extracted: ExtractedDocument, *, page_count: int, title: str | None, empty_pages: list[int]
) -> DocumentInfo:
    meta = {normalize(b.label or ""): b.text.split(":", 1)[-1].strip().rstrip(".")
            for b in extracted.blocks if b.kind == "metadata_row"}
    prose = [b for b in extracted.blocks if b.kind == "text"]
    text = "\n".join(b.text for b in prose)
    # Whole sentences (split on ". " + capital), so "v1.2" or ".pdf" are not truncated.
    # Change-log rows ("Se sustituye la bitácora en papel...") are not supersession statements.
    sentences = [s for b in prose for s in split_sentences(b.text)]
    effective = (
        _parse_date(meta.get("fecha de vigencia", ""), re.compile(_DATE, re.IGNORECASE))
        or _parse_effective_date(text)
    )
    code = meta.get("codigo de documento") or next(iter(_DOC_CODE.findall(text[:400])), None)
    references = sorted({c for c in _DOC_CODE.findall(extracted.full_text) if c != code})
    return DocumentInfo(
        document=extracted.document,
        title=title,
        pages=page_count,
        code=code,
        version=meta.get("version"),
        effective_date=effective,
        supersedes=[s for s in sentences if _SUPERSEDES_RE.search(s)],
        precedence_statements=[s for s in sentences if _PRECEDENCE_RE.search(s)],
        references=references,
        empty_pages=empty_pages,
    )


def discover_pdfs(pdf_dir: str | Path) -> list[Path]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        return []
    return sorted(p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
