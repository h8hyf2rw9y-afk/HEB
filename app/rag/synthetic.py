"""Synthetic policy PDFs for tests and demos.

These documents are FICTIONAL. They do not reproduce or approximate real H-E-B
policies; they only exercise extraction, sections, multi-page citations,
supersession, unresolved conflicts and abstention.
"""

from __future__ import annotations

from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore[no-redef]

DISCLAIMER = "DOCUMENTO SINTÉTICO DE PRUEBA. No representa políticas reales de H-E-B."

# Each document: title, list of pages; each page: list of (heading | None, paragraph).
SYNTHETIC_DOCS: dict[str, dict] = {
    "SINTETICO_politica_devoluciones_v1.pdf": {
        "title": "Política de Devoluciones v1",
        "pages": [
            [
                (None, DISCLAIMER + " Vigente a partir del 1 de enero de 2023."),
                ("1. Objetivo", "Establecer las reglas para aceptar devoluciones de mercancía en tienda."),
                (
                    "2. Plazo de devolución",
                    "El cliente podrá devolver productos en un plazo de 30 días naturales posteriores "
                    "a la compra presentando su ticket. Sin ticket solo se ofrece cambio por el mismo producto.",
                ),
            ],
            [
                (
                    "3. Productos excluidos",
                    "No se aceptan devoluciones de productos perecederos abiertos ni de tarjetas de regalo.",
                ),
                (
                    "4. Reembolsos",
                    "El reembolso se realizará al método de pago original en un máximo de 5 días hábiles.",
                ),
            ],
        ],
    },
    "SINTETICO_politica_devoluciones_v2.pdf": {
        "title": "Política de Devoluciones v2",
        "pages": [
            [
                (
                    None,
                    DISCLAIMER + " Vigente a partir del 1 de julio de 2024. Este documento reemplaza a "
                    "SINTETICO_politica_devoluciones_v1.pdf.",
                ),
                (
                    "2. Plazo de devolución",
                    "El cliente podrá devolver productos en un plazo de 15 días naturales posteriores "
                    "a la compra presentando su ticket.",
                ),
                (
                    "3. Electrónica",
                    "Los artículos electrónicos deben devolverse con su empaque original y todos sus accesorios.",
                ),
            ]
        ],
    },
    "SINTETICO_manual_seguridad_tienda.pdf": {
        "title": "Manual de Seguridad en Tienda",
        "pages": [
            [
                (None, DISCLAIMER),
                (
                    "1. Montacargas",
                    "Solo el personal certificado puede operar montacargas. Está prohibido operar "
                    "montacargas sin chaleco reflejante.",
                ),
                (
                    "2. Derrames",
                    "Todo derrame en piso de venta debe señalizarse en un máximo de 5 minutos y "
                    "limpiarse de inmediato.",
                ),
            ],
            [
                (
                    "3. Equipo de protección personal",
                    "El personal de recibo debe usar guantes y calzado de seguridad durante toda la jornada.",
                ),
            ],
        ],
    },
    "SINTETICO_lineamientos_inventario.pdf": {
        "title": "Lineamientos de Inventario",
        "pages": [
            [
                (None, DISCLAIMER + " Vigente a partir del 1 de marzo de 2024."),
                ("1. Conteo cíclico", "El conteo cíclico de inventario se realiza cada semana por departamento."),
                (
                    "2. Ajustes de inventario",
                    "Los ajustes de inventario mayores al 5% requieren aprobación del gerente de tienda.",
                ),
            ]
        ],
    },
    "SINTETICO_manual_operativo.pdf": {
        "title": "Manual Operativo de Tienda",
        "pages": [
            [
                (None, DISCLAIMER),
                (
                    "4. Ajustes de inventario",
                    "Los ajustes de inventario mayores al 10% requieren aprobación del gerente de tienda.",
                ),
                ("5. Apertura", "La tienda debe abrir con al menos dos cajas operando."),
            ]
        ],
    },
}


def protect_directory(directory: Path) -> None:
    """Write a ``.gitignore`` containing ``*`` so Git ignores the folder's contents."""
    gitignore = directory / ".gitignore"
    if not gitignore.exists() or gitignore.read_text(encoding="utf-8").strip() != "*":
        gitignore.write_text("*\n", encoding="utf-8")


def _write_pdf(path: Path, title: str, pages: list[list[tuple[str | None, str]]]) -> None:
    doc = pymupdf.open()
    for page_no, blocks in enumerate(pages):
        page = doc.new_page(width=612, height=792)
        y = 72.0
        if page_no == 0:
            page.insert_text((72, y), title, fontname="hebo", fontsize=18)
            y += 36
        for heading, paragraph in blocks:
            if heading:
                page.insert_text((72, y), heading, fontname="hebo", fontsize=13)
                y += 22
            rect = pymupdf.Rect(72, y, 540, y + 200)
            page.insert_textbox(rect, paragraph, fontname="helv", fontsize=11)
            lines = max(1, int(len(paragraph) / 85) + 1)
            y += lines * 14 + 18
    doc.set_metadata({"title": title, "subject": "synthetic test document"})
    doc.save(path)
    doc.close()


def write_synthetic_corpus(target_dir: str | Path) -> list[Path]:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    # Generated PDFs must never be committed by accident (same rule as the index):
    # the folder ignores itself regardless of the repository's root .gitignore.
    protect_directory(target)
    paths = []
    for name, spec in SYNTHETIC_DOCS.items():
        path = target / name
        _write_pdf(path, spec["title"], spec["pages"])
        paths.append(path)
    return paths
