"""Grounded cross-domain routing for store catalogue, source discovery and personnel gaps.

Never present unrelated PDF passages as labor policy. This module does not
generate employment advice and never uses the LLM for missing policy evidence.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "candidatos" / "datos"
POLICY_DIR = ROOT / "candidatos" / "politicas"


def plain(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


@dataclass(frozen=True)
class RoutedAnswer:
    answer: str
    status: str
    evidence: list[dict]


_STORE_COUNT = re.compile(
    r"\b(cuant[oa]s?|numero|cantidad|total)\b.{0,60}\b(tiendas?|sucursales?)\b"
    r"|\b(tiendas?|sucursales?)\b.{0,60}\b(tengo|tenemos|hay|existen|registradas)\b"
)
_SOURCE_DISCOVERY = re.compile(
    r"\b(que|cuales|lista|listar|muestra|mostrar|dime)\b.{0,65}"
    r"\b(documentos?|archivos?|fuentes|datos|informacion)\b"
    r"|\b(documentos?|archivos?|fuentes)\b.{0,35}\b(disponibles|tienes|manejas|consultas|puedes)\b"
)
_HR_SUBJECT = re.compile(
    r"\b(colaborador(?:es)?|emplead[oa]s?|trabajador(?:es)?|personal|"
    r"ausentismo|incapacidad|faltas?|inasistencias?)\b"
)
_HR_ABSENCE = re.compile(
    r"\b(falt[ao]|faltas?|ausent[ea]s?|ausencia|ausentismo|inasistencia|incapacidad|"
    r"no (?:ha|puede|pudo|podido|se ha)?\s*(?:presentad[oa]|venido|ir|asistid[oa]|asistir)|"
    r"sin (?:poder )?ir|no (?:ha podido|puede|pudo) ir|sin presentarse|no se presenta)\b"
)
_LONG_DURATION = re.compile(
    r"\b(semanas?|mes(?:es)?|dias?|varios|muchos|prolongad[ao]|seguido|continu[ao]|"
    r"consecutiv[ao]s?)\b"
)


def _store_answer(data_dir: Path) -> RoutedAnswer:
    path = data_dir / "tiendas.csv"
    evidence = [{"source_type": "csv", "source": "tiendas.csv", "reference": "tienda_id, nombre"}]
    if not path.is_file():
        return RoutedAnswer("No está disponible tiendas.csv para contar las tiendas.", "insufficient_data", [])
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        unique: dict[str, str] = {}
        for row in rows:
            code, name = (row.get("tienda_id") or "").strip(), (row.get("nombre") or "").strip()
            if code and name:
                unique[code] = name
        if not unique:
            return RoutedAnswer("tiendas.csv no contiene tiendas válidas.", "insufficient_data", [])
        details = "\n".join(f"- {code}: {name}" for code, name in sorted(unique.items()))
        return RoutedAnswer(
            f"En el catálogo proporcionado hay {len(unique)} tiendas registradas:\n{details}",
            "answered", evidence
        )
    except (OSError, UnicodeError, csv.Error):
        return RoutedAnswer("No pude leer el catálogo de tiendas en este momento.", "error", [])


def _source_answer(data_dir: Path, policy_dir: Path) -> RoutedAnswer:
    datasets = [
        ("ventas.csv", "ventas de junio a agosto de 2026"),
        ("inventario.csv", "existencias y puntos de reorden en el corte disponible"),
        ("tickets_mesa_servicio.csv", "tickets de soporte y su estado"),
        ("catalogo_productos.csv", "SKUs y productos"),
        ("tiendas.csv", "catálogo de tiendas"),
    ]
    known_csv = [(name, desc) for name, desc in datasets if (data_dir / name).is_file()]
    pdfs = sorted(p.name for p in policy_dir.glob("*.pdf") if p.is_file()) if policy_dir.is_dir() else []
    if not known_csv and not pdfs:
        return RoutedAnswer("No encontré fuentes CSV o PDF disponibles en esta instalación.", "insufficient_data", [])
    lines = ["Puedo consultar estas fuentes disponibles:"]
    if known_csv:
        lines.append("Datos CSV:")
        lines.extend(f"- {name}: {description}." for name, description in known_csv)
    if pdfs:
        lines.append("Documentos PDF de políticas:")
        lines.extend(f"- {name}" for name in pdfs)
    lines.append("Puedo responder preguntas sobre esos archivos e indicar la fuente. "
                 "No puedo entregar documentos laborales o políticas que no estén cargados.")
    evidence = (
        [{"source_type": "csv", "source": name, "reference": "archivo disponible"} for name, _ in known_csv]
        + [{"source_type": "pdf", "source": name, "reference": "documento disponible"} for name in pdfs]
    )
    return RoutedAnswer("\n".join(lines), "answered", evidence)


def route_business_question(
    question: str, *,
    data_dir: Path = DATA_DIR,
    policy_dir: Path = POLICY_DIR,
) -> RoutedAnswer | None:
    q = plain(question).strip(" ¡!¿?.,;:")
    if _STORE_COUNT.search(q):
        return _store_answer(data_dir)
    if _SOURCE_DISCOVERY.search(q) and not re.search(
        r"\b(establece|dice|segun|sobre|acerca|respecto|devolucion|reembolso|merma|"
        r"caducidad|inventario|venta|salario|colaborador|empleado)\b", q
    ):
        return _source_answer(data_dir, policy_dir)
    # Long-lasting employment absence is NOT covered by the one-shift staffing
    # contingency in FAQ P25. Do not pass it to a PDF retriever that may return
    # unrelated reimbursement or store-opening policy.
    if _HR_SUBJECT.search(q) and _HR_ABSENCE.search(q) and (
        _LONG_DURATION.search(q) or "lleva" in q or "desde hace" in q
    ):
        return RoutedAnswer(
            "Entiendo que un colaborador lleva tiempo sin asistir. "
            "Los documentos disponibles no establecen un procedimiento verificado "
            "para resolver ausencias prolongadas, incapacidades o medidas laborales; "
            "no sería correcto recomendar una sanción o terminación con esas fuentes. "
            "¿La ausencia está respaldada por una incapacidad o permiso, o aún no conocen el motivo? "
            "Con ese dato puedes plantear el caso al área de Recursos Humanos para revisar "
            "el Reglamento Interior de Trabajo y el procedimiento vigente.",
            "needs_clarification", []
        )
    return None
