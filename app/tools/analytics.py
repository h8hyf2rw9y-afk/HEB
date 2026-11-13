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
# Policy questions mentioning operational terms must never be mistaken for CSV metrics.
# Keep these safeguards narrow: ticket means sales receipt in a returns policy,
# inventory counting frequency is an operating procedure, and merchandise
# holding time after recording shrinkage is a policy question.
POLICY_QUESTION_GUARDS = (
    re.compile(r"\b(devolucion(?:es)?|devolver|reembolso)\b.*\bsin ticket\b"),
    re.compile(r"\b(frecuencia|cada cuanto|periodicidad)\b.*\b(cuenta|conteo|contar|inventario)\b"),
    re.compile(r"\b(cuanto tiempo|plazo)\b.*\b(permanecer|mermado|merma)\b"),
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
    if any(pattern.search(q) for pattern in POLICY_QUESTION_GUARDS):
        return False
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



def _compare_two_store_products(question: str, data_dir: Path) -> AnalyticalAnswer | None:
    """Compare top selling products across TWO explicitly named stores.

    This is a parameterized SQL template, never model-supplied SQL. If the
    intent is not explicit, leave it to the usual analytics router.
    """
    q = plain(question)
    if not re.search(r"\b(?:compara\w*|comparacion|versus|vs)\b", q):
        return None
    if not re.search(r"\b(?:productos?|sku)\b", q):
        return None
    stores_path = data_dir / "tiendas.csv"
    sales_path = data_dir / "ventas.csv"
    products_path = data_dir / "catalogo_productos.csv"
    if not all(p.is_file() for p in (stores_path, sales_path, products_path)):
        return _missing(question, "Faltan los CSV de ventas, tiendas o productos.")
    with stores_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matched = [
        item for item in rows
        if plain(item.get("nombre", "")).removeprefix("tienda ") in q
        or re.search(r"\b" + re.escape(plain(item.get("tienda_id", ""))) + r"\b", q)
    ]
    if len(matched) != 2:
        return None
    if re.search(r"\b20\d{2}\b", q) and any(year != "2026" for year in re.findall(r"\b20\d{2}\b", q)):
        return _missing(question, "Las ventas disponibles abarcan junio a agosto de 2026.")
    start, end, period = _period(question)
    if period == "multiple":
        return _missing(question, "Especifica un solo mes para esta comparación.")
    if re.search(r"\b(septiembre|octubre|noviembre|diciembre|enero|febrero|marzo|abril|mayo)\b", q):
        return _missing(question, "Las ventas disponibles abarcan junio a agosto de 2026.")
    import duckdb

    sql = """
        SELECT t.nombre, p.nombre, SUM(v.venta_neta_mxn) AS venta
        FROM read_csv_auto(?) v
        JOIN read_csv_auto(?) t ON v.tienda_id = t.tienda_id
        JOIN read_csv_auto(?) p ON v.sku = p.sku
        WHERE v.fecha >= CAST(? AS DATE) AND v.fecha < CAST(? AS DATE)
          AND v.tienda_id IN (?, ?)
        GROUP BY t.nombre, p.nombre
        ORDER BY t.nombre, venta DESC, p.nombre
    """
    with duckdb.connect() as con:
        results = con.execute(sql, [
            str(sales_path), str(stores_path), str(products_path),
            start, end, matched[0]["tienda_id"], matched[1]["tienda_id"],
        ]).fetchall()
    lines = []
    for item in matched:
        name = item["nombre"]
        ranked = [(product, amount) for store, product, amount in results if store == name][:5]
        lines.append(f"{name}:")
        lines.extend(f"- {product}: {_money(amount)}" for product, amount in ranked)
        if not ranked:
            lines.append("- No hay ventas de productos registradas para el periodo.")
    by_store = {
        name: {product: float(amount) for tienda, product, amount in results if tienda == name}
        for name in (matched[0]["nombre"], matched[1]["nombre"])
    }
    first, second = matched[0]["nombre"], matched[1]["nombre"]
    products_all = set(by_store[first]) | set(by_store[second])
    differences = sorted(
        products_all,
        key=lambda product: (-abs(by_store[first].get(product, 0.0) -
                                  by_store[second].get(product, 0.0)), product),
    )[:5]
    if differences:
        lines.append(f"Mayores diferencias por producto ({first} menos {second}):")
        for product in differences:
            a = by_store[first].get(product, 0.0)
            b = by_store[second].get(product, 0.0)
            lines.append(f"- {product}: {_money(a)} frente a {_money(b)}; "
                         f"diferencia {_money(a - b)}.")
    return AnalyticalAnswer(
        f"Comparación de productos por ventas netas ({period}; máximo 5 por tienda):\n"
        + "\n".join(lines)
        + "\nEsta comparación describe ventas observadas; no demuestra sus causas.",
        "answered",
        [_source("ventas.csv", period), _source("tiendas.csv", "tienda_id"),
         _source("catalogo_productos.csv", "sku")],
        "sales_comparison",
    )

_WHY_LEAD = re.compile(r"\b(?:por que|a que se debe|que explica|como se explica)\b")
_LEAD_WORDS = re.compile(r"\b(?:mas|mayor|mejor|supero|lider|primer lugar|ventaja)\b")


def _signed_money(value: float) -> str:
    return ("-" if value < 0 else "") + _money(abs(value))


def _store_sales_drivers(question: str, data_dir: Path) -> AnalyticalAnswer | None:
    """«¿Por qué <tienda> vendió más que las otras en <mes>?» answered with what the
    CSVs can actually show: where the store's lead over the average of the other
    stores sits (volume vs. price per unit, product categories, channels).

    Parameterized SQL only. It describes observed differences and says plainly
    that the files do not record causes (promotions, location, traffic, marketing)."""
    q = plain(question)
    if not (_WHY_LEAD.search(q) and _LEAD_WORDS.search(q)):
        return None
    stores_path, sales_path, products_path = (data_dir / "tiendas.csv", data_dir / "ventas.csv",
                                              data_dir / "catalogo_productos.csv")
    store_id, store_name = _requested_store(question, stores_path)
    if not store_id or store_id in ("ambiguous", "unknown"):
        return None
    if not all(p.is_file() for p in (stores_path, sales_path, products_path)):
        return _missing(question, "Faltan los CSV de ventas, tiendas o productos.")
    start, end, period = _period(question)
    if period == "multiple":
        return _missing(question, "Especifica un solo mes para este análisis.")
    import duckdb

    base = """
        WITH v AS (SELECT * FROM read_csv_auto(?) WHERE fecha >= CAST(? AS DATE) AND fecha < CAST(? AS DATE))
    """
    with duckdb.connect() as con:
        totals = con.execute(base + """
            SELECT t.tienda_id, t.nombre, SUM(v.venta_neta_mxn), SUM(v.unidades)
            FROM v JOIN read_csv_auto(?) t ON t.tienda_id = v.tienda_id
            GROUP BY t.tienda_id, t.nombre ORDER BY 3 DESC, t.nombre
        """, [str(sales_path), start, end, str(stores_path)]).fetchall()
        by_category = con.execute(base + """
            SELECT p.categoria, v.tienda_id, SUM(v.venta_neta_mxn)
            FROM v JOIN read_csv_auto(?) p ON p.sku = v.sku
            GROUP BY p.categoria, v.tienda_id
        """, [str(sales_path), start, end, str(products_path)]).fetchall()
        by_channel = con.execute(base + """
            SELECT v.canal, v.tienda_id, SUM(v.venta_neta_mxn) FROM v GROUP BY v.canal, v.tienda_id
        """, [str(sales_path), start, end]).fetchall()
    if len(totals) < 2 or store_id not in {row[0] for row in totals}:
        return _missing(question, "No hay ventas suficientes de esa tienda en el periodo para compararla.")
    others = [row for row in totals if row[0] != store_id]
    rank = next(i for i, row in enumerate(totals, 1) if row[0] == store_id)
    sales = {row[0]: float(row[2]) for row in totals}
    units = {row[0]: float(row[3]) for row in totals}
    other_avg_sales = sum(sales[row[0]] for row in others) / len(others)
    other_avg_units = sum(units[row[0]] for row in others) / len(others)
    gap = sales[store_id] - other_avg_sales
    per_unit = sales[store_id] / units[store_id] if units[store_id] else 0.0
    other_per_unit = other_avg_sales / other_avg_units if other_avg_units else 0.0

    def gaps(rows) -> list[tuple[str, float, float, float]]:
        table: dict[str, dict[str, float]] = {}
        for key, tid, amount in rows:
            table.setdefault(key, {})[tid] = float(amount)
        out = []
        for key, per_store in table.items():
            mine = per_store.get(store_id, 0.0)
            avg = sum(per_store.get(row[0], 0.0) for row in others) / len(others)
            out.append((key, mine, avg, mine - avg))
        return sorted(out, key=lambda item: (-item[3], item[0]))

    category_gaps, channel_gaps = gaps(by_category), gaps(by_channel)
    lines = [f"Análisis de ventas netas de {store_name} frente al promedio de las otras {len(others)} tiendas ({period}).",
             "", "Lo que muestran los datos:"]
    if rank == 1:
        lines.append(f"- {store_name} fue la tienda con más ventas: {_money(sales[store_id])}, "
                     f"{_money(gap)} por encima del promedio de las otras ({_money(other_avg_sales)}).")
    else:
        lines.append(f"- En {period}, {store_name} no fue la tienda con más ventas: quedó en el lugar {rank} "
                     f"con {_money(sales[store_id])}; la primera fue {totals[0][1]} con {_money(float(totals[0][2]))}.")
    lines.append(f"- Volumen: vendió {units[store_id]:,.0f} unidades frente a un promedio de "
                 f"{other_avg_units:,.0f} en las otras tiendas.")
    lines.append(f"- Venta promedio por unidad: {_money(per_unit)} frente a {_money(other_per_unit)} en las otras tiendas.")
    top_categories = [c for c in category_gaps if c[3] > 0][:3]
    if top_categories:
        lines.append("- Categorías con mayor ventaja sobre el promedio de las otras tiendas:")
        lines.extend(f"  - {name}: {_money(mine)} frente a {_money(avg)} (diferencia {_signed_money(diff)})"
                     for name, mine, avg, diff in top_categories)
    lines.append("- Ventas por canal frente al promedio de las otras tiendas:")
    lines.extend(f"  - {name}: {_money(mine)} frente a {_money(avg)} (diferencia {_signed_money(diff)})"
                 for name, mine, avg, diff in channel_gaps)
    if rank == 1 and gap > 0 and top_categories:
        share = sum(c[3] for c in top_categories) / gap
        volume_driven = units[store_id] > other_avg_units and abs(per_unit - other_per_unit) <= 0.05 * other_per_unit
        lines += ["", "Lectura de los datos:",
                  f"- Las {len(top_categories)} categorías anteriores concentran el {share * 100:.1f}% de la ventaja "
                  "sobre el promedio de las otras tiendas."]
        lines.append("- La ventaja viene sobre todo de vender más unidades, con una venta por unidad similar."
                     if volume_driven else
                     "- La ventaja combina diferencias de volumen y de venta promedio por unidad.")
    lines += ["", "Lo que los datos no permiten afirmar:",
              "- Los CSV registran ventas, unidades, descuentos, canales y productos, pero no promociones, "
              "ubicación, tráfico de clientes ni campañas de marketing. Estas diferencias describen dónde se "
              "concentra la venta; no demuestran sus causas."]
    return AnalyticalAnswer(
        "\n".join(lines), "answered",
        [_source("ventas.csv", period), _source("tiendas.csv", "tienda_id"),
         _source("catalogo_productos.csv", "categoria")],
        "sales_drivers",
    )


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
    # Recognize multi-store comparison BEFORE one-store resolution.
    comparison = _compare_two_store_products(question, data_dir)
    if comparison is not None:
        return comparison
    drivers = _store_sales_drivers(question, data_dir)
    if drivers is not None:
        return drivers
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
    # Reject years outside the dataset rather than quietly using 2026 data.
    years = re.findall(r"\b20\d{2}\b", q)
    if any(year != "2026" for year in years):
        return _missing(question, "El archivo de ventas solo contiene junio a agosto de 2026.")
    if re.search(r"\b(mediana|margen|ganancia|utilidad|pronostico|proyeccion)\b", q):
        return _missing(question, "No existe una consulta verificada para esa métrica.")
    sku_match = re.search(r"\bsku-\d+\b", q)
    sku_value = sku_match.group(0).upper() if sku_match else None
    if sku_value:
        if not products.is_file():
            return _missing(question, "No está disponible catalogo_productos.csv.")
        with products.open(encoding="utf-8-sig", newline="") as source_file:
            if not any(row["sku"].upper() == sku_value for row in csv.DictReader(source_file)):
                return _missing(question, "El SKU indicado no existe en el catálogo.")
    start, end, period = _period(question)
    if period == "multiple":
        return _missing(question, "Especifica solo un mes: junio, julio o agosto de 2026.")
    if re.search(r"\b(septiembre|octubre|noviembre|diciembre|enero|febrero|marzo|abril|mayo|2025|2027)\b", q):
        return _missing(question, "Las ventas disponibles abarcan únicamente junio a agosto de 2026.")

    # Distinguish money from units. Net sales already account for refunds (negative rows).
    units = bool(re.search(r"\b(unidades|piezas|cantidad de productos)\b", q))
    discounts = bool(re.search(r"\b(descuento|descuentos)\b", q))
    count_transactions = bool(re.search(r"\b(cuantas ventas|numero de ventas|cantidad de ventas|numero de registros)\b", q))
    average = bool(re.search(r"\b(promedio|media)\b", q))
    measure = "v.unidades" if units else "v.descuento_mxn" if discounts else "v.venta_neta_mxn"
    metric_label = ("registros de venta" if count_transactions else "promedio de unidades" if average and units
                    else "promedio de ventas netas" if average else "unidades netas" if units
                    else "descuentos" if discounts else "ventas netas")
    aggregation = "COUNT(*)" if count_transactions else f"AVG({measure})" if average else f"SUM({measure})"
    ranking = bool(re.search(r"\b(mas|mayor|mejor|menos|menor|peor|ranking|top)\b", q))
    ascending = bool(re.search(r"\b(menos|menor|peor)\b", q))

    by_product = bool(re.search(r"\b(producto|productos|sku)\b", q)) and not bool(sku_value)
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
    if sku_value:
        # A named SKU is an exact filter, not a request to rank all products.
        if not re.search(r"\b(por tienda|por canal|por categoria)\b", q):
            group = None
    store_filter = " AND v.tienda_id = ?" if store_id else ""
    sku_filter = " AND v.sku = ?" if sku_value else ""
    if store_id:
        params.append(store_id)
    if sku_value:
        params.append(sku_value)
    if group:
        direction = "ASC" if ascending else "DESC"
        sql = f"""
            SELECT {group} AS nombre, {aggregation} AS valor
            FROM read_csv_auto(?) v JOIN read_csv_auto(?) t ON v.tienda_id = t.tienda_id
            {join}
            WHERE v.fecha >= CAST(? AS DATE) AND v.fecha < CAST(? AS DATE){store_filter}{sku_filter}
            GROUP BY 1 ORDER BY valor {direction}, nombre ASC LIMIT 5
        """
    else:
        sql = f"""
            SELECT {aggregation} AS valor
            FROM read_csv_auto(?) v JOIN read_csv_auto(?) t ON v.tienda_id = t.tienda_id
            {join}
            WHERE v.fecha >= CAST(? AS DATE) AND v.fecha < CAST(? AS DATE){store_filter}{sku_filter}
        """
    with duckdb.connect() as con:
        rows = con.execute(sql, params).fetchall()
    if not rows or rows[0][-1] is None:
        return _missing(question, "No hay filas de ventas para ese periodo.")

    def display(value):
        return f"{int(value):,} registros" if count_transactions else f"{float(value):,.2f} unidades" if average and units else f"{int(value):,} unidades" if units else _money(value)
    if group:
        lines = [f"{n}. {name}: {display(value)}" for n, (name, value) in enumerate(rows, start=1)]
        answer = f"{metric_label.capitalize()} por {group_name} ({period}), {'menor a mayor' if ascending else 'mayor a menor'}:\n" + "\n".join(lines)
    else:
        scope = (f" · {store_name}" if store_name else "") + (f" · {sku_value}" if sku_value else "")
        answer = f"Total de {metric_label} ({period}{scope}): {display(rows[0][0])}."
    evidence = [_source("ventas.csv", period), _source("tiendas.csv", "tienda_id")]
    if join_products or sku_value:
        evidence.append(_source("catalogo_productos.csv", "sku"))
    return AnalyticalAnswer(answer, "answered", evidence, "sales")
