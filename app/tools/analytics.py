"""Grounded Spanish analytical questions against the H-E-B exercise CSVs.

A small, intentional intent catalogue; never generate SQL from user input.
Unknown or ambiguous data questions abstain instead of inventing figures.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "candidatos" / "datos"
MONTHS = {"junio": "2026-06", "julio": "2026-07", "agosto": "2026-08"}
DATA_WORDS = re.compile(
    r"\b(venta|ventas|vendi|vendio|vendimos|vendieron|vender|venderemos|facturacion|ingresos|"
    r"unidades|producto|productos|sku|inventario|existencia|existencias|stock|"
    r"reorden|surtir|agotad[oa]s?|ticket|tickets|incidencias|mesa de servicio|"
    r"canal|categorias?|descuento|descuentos)\b"
)
POLICY_HINT = re.compile(
    r"\b(politica|procedimiento|manual|devolucion|devolver|reembolso|"
    r"merma|caducidad|caducado|faq|que hacer|como actuar|debo hacer|"
    r"se permite|puedo|plazo|regla|norma|protocolo)\b"
)
ANALYTIC_HINT = re.compile(
    r"\b(cuant[oa]s?|total|sum[ao]|mayor|menor|mas|menos|"
    r"mejor|peor|ranking|compar|promedio|por tienda|por sucursal|"
    r"en junio|en julio|en agosto|vend|vender|factur|"
    r"hay|tenemos|quedan|abiertos|pendientes|criticos)\b"
)


def plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def looks_like_data(question: str) -> bool:
    q = plain(question)
    if POLICY_HINT.search(q) and not re.search(r"\b(ventas|vendimos|vendio|facturacion|stock|inventario|tickets)\b", q):
        return False
    return bool(DATA_WORDS.search(q) and (ANALYTIC_HINT.search(q) or
        re.search(r"\b(inventario|existencias|stock|tickets|facturacion|ventas|reorden|agotad[oa]s?)\b", q)))


@dataclass(frozen=True)
class AnalyticalAnswer:
    answer: str
    status: str
    evidence: list[dict]
    intent: str | None = None


def _source(file: str, detail: str) -> dict[str, str]:
    return {"source_type": "csv", "source": file, "reference": detail}


def _money(value: float | int | None) -> str:
    return f"${value:,.2f} MXN" if value is not None else "sin dato"


def _period(question: str) -> tuple[str | None, str | None, str]:
    q = plain(question)
    found = [key for key in MONTHS if re.search(rf"\b{key}\b", q)]
    if len(found) > 1:
        return None, None, "multiple"
    if found:
        key = found[0]
        return MONTHS[key] + "-01", {"junio": "2026-07-01", "julio": "2026-08-01", "agosto": "2026-09-01"}[key], key + " de 2026"
    return "2026-06-01", "2026-09-01", "junio–agosto de 2026"


def _requested_store(question: str, stores_path: Path) -> tuple[str | None, str | None]:
    """Match explicitly named stores against the supplied store catalogue."""
    q = plain(question)
    if not stores_path.is_file():
        return None, None
    with stores_path.open(encoding="utf-8-sig", newline="") as f:
        known = list(csv.DictReader(f))
    matches = []
    for store in known:
        code = store["tienda_id"].lower()
        full = plain(store["nombre"])
        short = full.removeprefix("tienda ")
        if re.search(rf"\b{re.escape(code)}\b", q) or full in q or short in q:
            matches.append(store)
    if len(matches) == 1:
        return matches[0]["tienda_id"], matches[0]["nombre"]
    if len(matches) > 1:
        return "ambiguous", None
    if re.search(r"\bt\d{2}\b", q):
        return "unknown", None
    return None, None


def _missing(question: str, reason: str) -> AnalyticalAnswer:
    return AnalyticalAnswer(
        "No puedo obtener una cifra confiable de los CSV para esa pregunta. " + reason,
        "insufficient_data", [],
    )


def answer_data_question(question: str, data_dir: Path = DATA_DIR) -> AnalyticalAnswer:
    """Answer a narrow set of questions with read-only, parameterized DuckDB SQL."""
    q = plain(question)
    if not looks_like_data(question):
        return _missing(question, "La pregunta no corresponde a una consulta analítica identificable.")
    import duckdb

    sales = data_dir / "ventas.csv"
    stores = data_dir / "tiendas.csv"
    products = data_dir / "catalogo_productos.csv"
    inventory = data_dir / "inventario.csv"
    tickets = data_dir / "tickets_mesa_servicio.csv"
    if re.search(r"\b(precio|precios|cuesta|costo|costos|salario|nomina)\b", q):
        return _missing(question, "Esa métrica no está implementada en las consultas analíticas de esta primera versión.")
    store_id, store_name = _requested_store(question, stores)
    if store_id in ("ambiguous", "unknown"):
        return _missing(question, "No se identificó una sola tienda válida del catálogo.")

    if re.search(r"\b(inventario|existencias|existencia|stock|reorden|surtir|agotad[oa]s?)\b", q):
        if not all(path.is_file() for path in (inventory, stores, products)):
            return _missing(question, "No están disponibles los CSV de inventario, tiendas y catálogo.")
        # The snapshot is not a historical inventory series.
        if any(month in q for month in MONTHS):
            return _missing(question, "Solo existe un corte de inventario al 1 de septiembre de 2026.")
        # When the question asks which items need replenishment, provide
        # individual SKUs and quantities rather than just a summary count.
        listing = bool(re.search(r"\b(que|cuales|lista|listar|detalle|productos|sku)\b", q)) and (
            "reorden" in q or "agotad" in q or "surtir" in q
        )
        params = [str(inventory), str(stores), str(products)]
        where = "WHERE i.existencia <= i.punto_reorden" if "reorden" in q or "surtir" in q else "WHERE i.existencia = 0"
        if listing:
            if store_id:
                where += " AND i.tienda_id = ?"
                params.append(store_id)
            sql = f"""
                SELECT t.nombre, i.sku, p.nombre, i.existencia, i.punto_reorden
                FROM read_csv_auto(?) i
                JOIN read_csv_auto(?) t ON t.tienda_id = i.tienda_id
                JOIN read_csv_auto(?) p ON p.sku = i.sku
                {where}
                ORDER BY t.nombre, i.sku LIMIT 30
            """
            with duckdb.connect() as con:
                rows = con.execute(sql, params).fetchall()
            if not rows:
                return _missing(question, "No hay productos que cumplan la condición para esa tienda.")
            lines = [
                f"{store} · {sku} · {name}: existencia {current}, punto de reorden {threshold}."
                for store, sku, name, current, threshold in rows
            ]
            return AnalyticalAnswer(
                "Inventario al 1 de septiembre de 2026 (máximo 30 resultados):\n" + "\n".join(lines),
                "answered", [_source("inventario.csv", "corte 2026-09-01"),
                             _source("catalogo_productos.csv", "sku"),
                             _source("tiendas.csv", "tienda_id")], "inventory",
            )
        sql = """
            SELECT t.nombre, count(*) AS productos,
                   sum(CASE WHEN i.existencia <= i.punto_reorden THEN 1 ELSE 0 END) AS bajo_reorden,
                   sum(CASE WHEN i.existencia = 0 THEN 1 ELSE 0 END) AS agotados
            FROM read_csv_auto(?) i
            JOIN read_csv_auto(?) t ON t.tienda_id = i.tienda_id
            GROUP BY t.nombre ORDER BY t.nombre
        """
        with duckdb.connect() as con:
            rows = con.execute(sql, [str(inventory), str(stores)]).fetchall()
        if store_name:
            rows = [row for row in rows if row[0] == store_name]
        if not rows:
            return _missing(question, "El inventario no contiene registros para esa tienda.")
        lines = [
            f"{store}: {low} SKU en o por debajo del punto de reorden; {zero} sin existencia (de {total} SKU)."
            for store, total, low, zero in rows
        ]
        return AnalyticalAnswer(
            "Inventario al 1 de septiembre de 2026:\n" + "\n".join(lines),
            "answered", [_source("inventario.csv", "corte 2026-09-01"),
                         _source("tiendas.csv", "tienda_id")], "inventory",
        )

    if re.search(r"\b(ticket|tickets|incidencias|mesa de servicio)\b", q):
        if not all(path.is_file() for path in (tickets, stores)):
            return _missing(question, "No están disponibles tickets_mesa_servicio.csv y tiendas.csv.")
        critical_only = bool(re.search(r"\b(critica|criticas|critico|criticos)\b", q))
        pending_only = bool(re.search(r"\b(abiert[oa]s?|pendientes?|sin resolver|en proceso)\b", q))
        conditions = []
        if critical_only:
            conditions.append("lower(s.prioridad) IN ('crítica', 'critica')")
        if pending_only:
            conditions.append("lower(s.estado) IN ('abierto', 'en proceso')")
        if store_id:
            conditions.append("s.tienda_id = ?")
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT t.nombre, COUNT(*) AS total
            FROM read_csv_auto(?) s
            JOIN read_csv_auto(?) t ON s.tienda_id = t.tienda_id
            {where}
            GROUP BY t.nombre ORDER BY t.nombre
        """
        params = [str(tickets), str(stores)] + ([store_id] if store_id else [])
        with duckdb.connect() as con:
            rows = con.execute(sql, params).fetchall()
        label = ("críticos " if critical_only else "") + ("pendientes" if pending_only else "registrados")
        if not rows:
            return AnalyticalAnswer(
                f"Tickets {label}, por tienda:\nTotal: 0 tickets.",
                "answered", [_source("tickets_mesa_servicio.csv", "prioridad y estado al corte"),
                             _source("tiendas.csv", "tienda_id")], "tickets",
            )
        lines = [f"{store}: {total} tickets {label}." for store, total in rows]
        return AnalyticalAnswer(
            f"Tickets {label}, por tienda:\n" + "\n".join(lines) +
            f"\nTotal: {sum(row[1] for row in rows)} tickets.",
            "answered", [_source("tickets_mesa_servicio.csv", "prioridad y estado al corte"),
                         _source("tiendas.csv", "tienda_id")], "tickets",
        )

    if not all(path.is_file() for path in (sales, stores)):
        return _missing(question, "No están disponibles ventas.csv y tiendas.csv.")
    start, end, period = _period(question)
    if period == "multiple":
        return _missing(question, "Especifica solo un mes: junio, julio o agosto de 2026.")
    if re.search(r"\b(septiembre|octubre|noviembre|diciembre|enero|febrero|marzo|abril|mayo|2025|2027)\b", q):
        return _missing(question, "Las ventas disponibles abarcan únicamente junio a agosto de 2026.")

    # Distinguish money from units. Net sales already account for refunds (negative rows).
    units = bool(re.search(r"\b(unidades|piezas|cantidad de productos)\b", q))
    discounts = bool(re.search(r"\b(descuento|descuentos)\b", q))
    measure = "v.unidades" if units else "v.descuento_mxn" if discounts else "v.venta_neta_mxn"
    metric_label = "unidades netas" if units else "descuentos" if discounts else "ventas netas"
    ranking = bool(re.search(r"\b(mas|mayor|mejor|menos|menor|peor|ranking|top)\b", q))
    ascending = bool(re.search(r"\b(menos|menor|peor)\b", q))

    by_product = bool(re.search(r"\b(producto|productos|sku)\b", q))
    by_category = bool(re.search(r"\b(categoria|categorias)\b", q))
    by_channel = bool(re.search(r"\b(canal|ecommerce|piso)\b", q))
    by_store = bool(re.search(r"\b(tienda|tiendas|sucursal|sucursales)\b", q))
    group = ("p.nombre" if by_product else "p.categoria" if by_category else
             "v.canal" if by_channel else "t.nombre" if by_store else None)
    group_name = "producto" if by_product else "categoría" if by_category else "canal" if by_channel else "tienda"
    if store_id and not ranking and not re.search(r"\b(por tienda|por sucursal|cada tienda|cada sucursal)\b", q):
        group = None
    if ranking and group is None:
        return _missing(question, "Indica si deseas comparar tiendas, productos, categorías o canales.")

    join_products = by_product or by_category
    if join_products and not products.is_file():
        return _missing(question, "No está disponible catalogo_productos.csv.")
    params = [str(sales), str(stores)]
    join = " JOIN read_csv_auto(?) p ON p.sku = v.sku" if join_products else ""
    if join_products:
        params.append(str(products))
    params.extend([start, end])
    store_filter = " AND v.tienda_id = ?" if store_id else ""
    if store_id:
        params.append(store_id)
    if group:
        direction = "ASC" if ascending else "DESC"
        sql = f"""
            SELECT {group} AS nombre, SUM({measure}) AS valor
            FROM read_csv_auto(?) v JOIN read_csv_auto(?) t ON v.tienda_id = t.tienda_id
            {join}
            WHERE v.fecha >= CAST(? AS DATE) AND v.fecha < CAST(? AS DATE){store_filter}
            GROUP BY 1 ORDER BY valor {direction}, nombre ASC LIMIT 5
        """
    else:
        sql = f"""
            SELECT SUM({measure}) AS valor
            FROM read_csv_auto(?) v JOIN read_csv_auto(?) t ON v.tienda_id = t.tienda_id
            {join}
            WHERE v.fecha >= CAST(? AS DATE) AND v.fecha < CAST(? AS DATE){store_filter}
        """
    with duckdb.connect() as con:
        rows = con.execute(sql, params).fetchall()
    if not rows or rows[0][-1] is None:
        return _missing(question, "No hay filas de ventas para ese periodo.")

    def display(value):
        return f"{int(value):,} unidades" if units else _money(value)
    if group:
        lines = [f"{n}. {name}: {display(value)}" for n, (name, value) in enumerate(rows, start=1)]
        answer = f"{metric_label.capitalize()} por {group_name} ({period}), {'menor a mayor' if ascending else 'mayor a menor'}:\n" + "\n".join(lines)
    else:
        scope = f" · {store_name}" if store_name else ""
        answer = f"Total de {metric_label} ({period}{scope}): {display(rows[0][0])}."
    evidence = [_source("ventas.csv", period), _source("tiendas.csv", "tienda_id")]
    if join_products:
        evidence.append(_source("catalogo_productos.csv", "sku"))
    return AnalyticalAnswer(answer, "answered", evidence, "sales")
