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
    r"\b(?:cuantas?|cuantos?|numero|cantidad|total)\s+(?:de\s+)?(?:las\s+)?"
    r"(?:tiendas?|sucursales?)\b"
    r"|\b(?:tiendas?|sucursales?)\b.{0,40}\b(?:tengo|tenemos|hay|existen|registradas)\b"
)
_PRODUCT_COUNT = re.compile(
    r"\b(?:cuantos?|numero|cantidad|total)\s+(?:de\s+)?(?:productos?|sku)\b"
    r".{0,70}\b(?:cada|por|en)\s+(?:una?\s+)?(?:tiendas?|sucursales?)\b"
    r"|\b(?:cuantos?|numero|cantidad|total)\s+(?:de\s+)?(?:sku|productos?)\b"
    r".{0,70}\b(?:tiene|hay)\b.{0,20}\b(?:tiendas?|sucursales?)\b"
)

def _product_count_answer(data_dir: Path) -> RoutedAnswer:
    """Count distinct inventory SKUs per store at the documented snapshot.

    A catalogued product is not necessarily stocked. Keep this count scoped
    explicitly to the inventory snapshot and avoid discussing reorders.
    """
    inventory, stores = data_dir / "inventario.csv", data_dir / "tiendas.csv"
    if not inventory.is_file() or not stores.is_file():
        return RoutedAnswer(
            "No están disponibles inventario.csv y tiendas.csv para contar SKU por tienda.",
            "insufficient_data", [],
        )
    try:
        with stores.open(encoding="utf-8-sig", newline="") as handle:
            store_rows = list(csv.DictReader(handle))
        with inventory.open(encoding="utf-8-sig", newline="") as handle:
            inventory_rows = list(csv.DictReader(handle))
        names = {r["tienda_id"]: r["nombre"] for r in store_rows
                 if r.get("tienda_id") and r.get("nombre")}
        sku_counts = {code: set() for code in names}
        for row in inventory_rows:
            code, sku = row.get("tienda_id", ""), row.get("sku", "")
            if code in sku_counts and sku:
                sku_counts[code].add(sku)
        if not names:
            return RoutedAnswer("No hay tiendas válidas en el catálogo.", "insufficient_data", [])
        lines = [f"- {names[code]}: {len(sku_counts[code])} SKU registrados."
                 for code in sorted(names)]
        return RoutedAnswer(
            "Productos distintos por tienda según inventario.csv "
            "(corte del 1 de septiembre de 2026):\n" + "\n".join(lines)
            + "\nEsto cuenta SKU registrados, no productos disponibles para venta "
            "ni unidades en existencia.",
            "answered",
            [{"source_type": "csv", "source": "inventario.csv",
              "reference": "SKU por tienda; corte 2026-09-01"},
             {"source_type": "csv", "source": "tiendas.csv", "reference": "tienda_id"}],
        )
    except (OSError, UnicodeError, csv.Error, KeyError):
        return RoutedAnswer("No pude contar los SKU del inventario.", "error", [])


_SOURCE_DISCOVERY = re.compile(
    r"\b(que|cuales|lista|listar|muestra|mostrar|dime)\b.{0,65}"
    r"\b(documentos?|archivos?|fuentes|datos|informacion)\b"
    r"|\b(documentos?|archivos?|fuentes)\b.{0,35}\b(disponibles|tienes|manejas|consultas|puedes)\b"
)
_HR_SUBJECT = re.compile(
    r"\b(colaborador(?:es)?|emplead[oa]s?|trabajador(?:es)?|personal|"
    r"ausentismo|incapacidad|incapacitad[oa]s?|faltas?|inasistencias?)\b"
)
_HR_ABSENCE = re.compile(
    r"\b(falt[ao]|faltas?|ausent[ea]s?|ausencia|ausentismo|inasistencia|incapacidad|"
    r"no (?:ha|puede|pudo|podido|se ha)?\s*(?:presentad[oa]|venido|ir|asistid[oa]|asistir)|"
    r"sin (?:poder )?ir|no (?:ha podido|puede|pudo) ir|incapacitad[oa]s?|sin presentarse|no se presenta)\b"
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
    if _PRODUCT_COUNT.search(q) and not re.search(r"\b(?:ventas?|vend\w*|reorden|agotad\w*|stock bajo)\b", q):
        return _product_count_answer(data_dir)
    if _SOURCE_DISCOVERY.search(q) and not re.search(
        r"\b(establece|dice|segun|sobre|acerca|respecto|devolucion|reembolso|merma|"
        r"caducidad|inventario|venta|salario|colaborador|empleado)\b", q
    ):
        return _source_answer(data_dir, policy_dir)
    # Long-lasting employment absence is NOT covered by the one-shift staffing
    # contingency in FAQ P25. Do not pass it to a PDF retriever that may return
    # unrelated reimbursement or store-opening policy.
    employment_decision = bool(re.search(
        r"\b(desped\w*|despid\w*|despido|sancion\w*|baja|"
        r"rescisi\w*|contrato|laboral)\b", q
    ))
    explicit_absence = bool(_HR_ABSENCE.search(q) or re.search(r"\bfaltar\b", q))
    employee_context = bool(_HR_SUBJECT.search(q) or (employment_decision and explicit_absence))
    if employee_context and explicit_absence and (
        _LONG_DURATION.search(q) or "lleva" in q or "desde hace" in q
        or employment_decision
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
