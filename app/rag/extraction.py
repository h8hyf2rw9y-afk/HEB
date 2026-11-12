"""PDF text extraction with PyMuPDF, preserving page numbers and section headings."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .models import DocumentInfo
from .text import split_sentences

try:  # PyMuPDF >= 1.24 exposes ``pymupdf``; older releases only ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - depends on installed version
    import fitz as pymupdf  # type: ignore[no-redef]


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


_NUMBERED_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?)|(?:[IVXLC]+\.)|(?:secci[oó]n|cap[ií]tulo|art[ií]culo|anexo)\b)\s*\S",
    re.IGNORECASE,
)
_BOLD_FLAG = 1 << 4


@dataclass
class TextLine:
    text: str
    size: float
    bold: bool


@dataclass
class PageBlock:
    """A paragraph-like block of text on a page, tagged with its active section."""

    page: int
    section: str | None
    text: str


@dataclass
class ExtractedDocument:
    document: str
    path: Path
    blocks: list[PageBlock] = field(default_factory=list)
    info: DocumentInfo | None = None

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


def _is_heading(line: TextLine, body_size: float) -> bool:
    text = line.text.strip()
    words = text.split()
    if not words or len(words) > 14 or len(text) < 3:
        return False
    if text.endswith((".", ",", ";")) and not _NUMBERED_HEADING.match(text):
        return False
    larger = line.size >= body_size * 1.15
    numbered = bool(_NUMBERED_HEADING.match(text))
    return larger or (line.bold and (numbered or text[:1].isupper()))


def _page_lines(page: "pymupdf.Page") -> list[list[TextLine]]:
    """Return text lines grouped by PyMuPDF block."""
    out: list[list[TextLine]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        lines: list[TextLine] = []
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = " ".join(s["text"].strip() for s in spans)
            size = max(s.get("size", 0.0) for s in spans)
            bold = all((s.get("flags", 0) & _BOLD_FLAG) or "bold" in s.get("font", "").lower() for s in spans)
            lines.append(TextLine(text=text, size=size, bold=bool(bold)))
        if lines:
            out.append(lines)
    return out


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
        pages = [_page_lines(page) for page in doc]
        page_count = doc.page_count

    sizes = [ln.size for page in pages for blk in page for ln in blk for _ in range(max(1, len(ln.text)))]
    body_size = statistics.median(sizes) if sizes else 11.0

    result = ExtractedDocument(document=path.name, path=path)
    empty_pages: list[int] = []
    section: str | None = None
    title: str | None = None
    for page_no, blocks in enumerate(pages, start=1):
        if not blocks:
            empty_pages.append(page_no)
            continue
        for lines in blocks:
            buffer: list[str] = []
            for line in lines:
                if _is_heading(line, body_size):
                    if buffer:
                        result.blocks.append(PageBlock(page_no, section, " ".join(buffer)))
                        buffer = []
                    if title is None and line.size >= body_size * 1.3:
                        title = line.text.strip()
                    section = line.text.strip()
                else:
                    buffer.append(line.text.strip())
            if buffer:
                result.blocks.append(PageBlock(page_no, section, " ".join(buffer)))

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
_EFFECTIVE_RE = re.compile(
    r"(?:vigente|vigencia|entra en vigor|en vigor|efectiv[oa])[^.\n]{0,40}?"
    r"(?:(\d{1,2})\s+de\s+([a-záéíóú]+)\s+(?:de|del)\s+(\d{4})|(\d{4})-(\d{2})-(\d{2})|(\d{1,2})/(\d{1,2})/(\d{4}))",
    re.IGNORECASE,
)
_SUPERSEDES_RE = re.compile(
    r"\b(?:reemplaza|sustituye|deja sin efecto|deroga|actualiza y reemplaza)\b", re.IGNORECASE
)
_PRECEDENCE_RE = re.compile(
    r"\ben caso de (?:conflicto|discrepancia|contradicci[oó]n)\b", re.IGNORECASE
)


def _parse_effective_date(text: str) -> str | None:
    match = _EFFECTIVE_RE.search(text)
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


def build_document_info(
    extracted: ExtractedDocument, *, page_count: int, title: str | None, empty_pages: list[int]
) -> DocumentInfo:
    text = extracted.full_text
    # Whole sentences (split on ". " + capital), so "v1.2" or ".pdf" are not truncated.
    sentences = [s for b in extracted.blocks for s in split_sentences(b.text)]
    return DocumentInfo(
        document=extracted.document,
        title=title,
        pages=page_count,
        effective_date=_parse_effective_date(text),
        supersedes=[s for s in sentences if _SUPERSEDES_RE.search(s)],
        precedence_statements=[s for s in sentences if _PRECEDENCE_RE.search(s)],
        empty_pages=empty_pages,
    )


def discover_pdfs(pdf_dir: str | Path) -> list[Path]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        return []
    return sorted(p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
