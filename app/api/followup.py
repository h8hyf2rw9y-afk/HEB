"""Conservative continuation of analytics questions from verified session evidence.

Never reuse past assistant prose as a new fact: the result here is a QUERY
to run against the original CSVs again. Only source-backed session turns
may provide the omitted store and period. Unsupported ambiguity stays put.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

from .session_memory import VerifiedTurn

DATA_DIR = Path(__file__).resolve().parents[2] / "candidatos" / "datos"
MONTH = r"(?:junio|julio|agosto)"
MONTH_RE = re.compile(r"\b" + MONTH + r"\b")
SALES_RE = re.compile(r"\b(?:venta\w*|vend\w*|factur\w*)\b")
FOLLOWUP_RE = re.compile(
    r"\b(?:esa tienda|esa sucursal|esa misma tienda|le fue|sus ventas|"
    r"cuanto vendio|que productos|cuales productos|sus productos|"
    r"y en (?:junio|julio|agosto)|y cuanto|en (?:junio|julio|agosto)|por que|a que se debe|"
    r"compar\w*(?:la|lo)|compar\w* con)\b"
)


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in normalized if not unicodedata.combining(c))


EXPLICIT_DATA_REF = re.compile(
    r"\b(?:esa tienda|esa sucursal|esa misma tienda|sus ventas|sus productos|"
    r"compar\w*(?:la|lo)|le fue a esa)\b"
)


def _named_store(question: str, data_dir: Path) -> str | None:
    path = data_dir / "tiendas.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        names = [r.get("nombre", "").strip() for r in csv.DictReader(handle)]
    q = _plain(question)
    hits = [n for n in names if n and _plain(n) in q]
    return hits[0] if len(hits) == 1 else None


def _top_store(previous: VerifiedTurn, data_dir: Path) -> str | None:
    """The store the previous verified data turn was about: the one it named
    explicitly, else the first ranked store of the recomputed answer."""
    if not re.search(r"\b(?:tienda|sucursal)\b", _plain(previous.question)):
        return None
    explicit = _named_store(previous.question, data_dir)
    if explicit:
        from app.tools.analytics import answer_data_question
        if answer_data_question(previous.question, data_dir).status == "answered":
            return explicit
    # Recompute the previous analytical answer from the CSV. Even a model
    # redaction that passed validation must not become an authority on rank.
    from app.tools.analytics import answer_data_question
    canonical = answer_data_question(previous.question, data_dir)
    if canonical.status != "answered":
        return None
    # A re-computed total for a single explicitly named store remains a
    # valid anchor for the next drill-down, even if no ranking is printed.
    has_rank = bool(re.search(r"(?m)^\s*1\.\s+", canonical.answer))
    if not has_rank and not re.search(r"\b(?:tienda|sucursal)\b", _plain(canonical.answer)):
        return None
    path = data_dir / "tiendas.csv"
    if not path.is_file():
        return None
    top_line = next(
        (line for line in canonical.answer.splitlines() if re.match(r"^\s*1\.\s+", line)),
        "",
    ) if has_rank else canonical.answer
    with path.open(encoding="utf-8-sig", newline="") as handle:
        known = [row.get("nombre", "").strip() for row in csv.DictReader(handle)]
    matches = [name for name in known if name and _plain(name) in _plain(top_line)]
    return matches[0] if len(matches) == 1 else None


def resolve_data_followup(question: str, previous: VerifiedTurn | None,
                          data_dir: Path = DATA_DIR) -> str | None:
    """Return a fully specified analytical QUERY or None when ambiguous.

    This never replays the previous answer, and never transfers store context
    to unrelated subjects or company policies.
    """
    if previous is None or not previous.is_grounded or previous.kind != "grounded_data":
        return None
    if not any(e.get("source") == "ventas.csv" for e in previous.evidence):
        return None
    q, prior = _plain(question.strip()), _plain(previous.question)
    if not FOLLOWUP_RE.search(q):
        return None
    if re.search(r"\b(?:devoluc\w*|reembols\w*|politic\w*|ticket|inventario|stock|merma)\b", q):
        return None

    # Never silently override an explicit reference to another store.
    store = _top_store(previous, data_dir)
    stores_path = data_dir / "tiendas.csv"
    named = []
    if stores_path.is_file():
        with stores_path.open(encoding="utf-8-sig", newline="") as handle:
            names = [r.get("nombre", "").strip() for r in csv.DictReader(handle)]
        named = [name for name in names if name and _plain(name).removeprefix("tienda ") in q]
        refers_back = re.search(r"\b(?:sus|esa|aquella)\b|\bcompar\w*(?:la|lo)\b", q)
        if named and not (len(named) == 1 and store and
                          named[0] != store and refers_back and
                          re.search(r"\b(?:compar\w*|versus|vs)\b", q)):
            return None

    current_month = MONTH_RE.findall(q)
    previous_month = MONTH_RE.findall(prior)
    if len(current_month) > 1 or len(previous_month) > 1:
        return None
    month = current_month[0] if current_month else previous_month[0] if previous_month else None
    if not month:
        return None
    explicit_years = re.findall(r"\b20\d{2}\b", q)
    if explicit_years and any(year != "2026" for year in explicit_years):
        return None

    # "Why did it sell more than the others?" is not proof of causality: answer
    # with a fresh, verifiable breakdown of the store against the average of the
    # other stores (volume, price per unit, categories, channels). The lead may
    # be stated in the question itself, not only in the previous ranking turn.
    why = bool(re.search(r"\b(?:por que|a que se debe|que explica)\b", q))
    lead = re.search(r"\b(?:mas|mayor|mejor|ranking|supero)\b", q) or re.search(
        r"\b(?:mas|mayor|mejor|ranking)\b", prior)
    versus_one = re.search(r"\bque (?:otra|la otra|la segunda)\b|\bsegundo lugar\b", q)
    if store and why and lead and not versus_one and not named:
        return f"¿Por qué {store} vendió más que las otras tiendas en {month} de 2026?"
    # "...than another (store)?" keeps the two-store product comparison.
    if (store and why
            and re.search(r"\b(?:mas|mayor|mejor|ranking)\b", prior)):
        from app.tools.analytics import answer_data_question
        canonical = answer_data_question(previous.question, data_dir)
        second_line = next(
            (line for line in canonical.answer.splitlines()
             if re.match(r"^\s*2\.\s+", line)), ""
        )
        if stores_path.is_file():
            alternatives = [name for name in names
                            if name != store and _plain(name) in _plain(second_line)]
            if len(alternatives) == 1:
                return (f"Compara productos de {store} y {alternatives[0]} "
                        f"en {month} de 2026.")

    if (len(named) == 1 and store and named[0] != store
            and re.search(r"\b(?:compar\w*|versus|vs)\b", q)
            and (re.search(r"\b(?:productos?|sku)\b", q) or re.search(r"\bcompar\w*(?:la|lo)\b", q))):
        return f"Compara productos de {store} y {named[0]} en {month} de 2026."

    # Short "¿Y en julio?" retains the previous ranking question, not
    # an arbitrary store; it is re-executed for the new month.
    if re.fullmatch(r"(?:y\s+)?en\s+" + MONTH + r"(?:\s+de\s+2026)?", q.strip(" ¿?¡!.,;:")):
        if not previous_month:
            return None
        return re.sub(r"\b" + previous_month[0] + r"\b", month, previous.question,
                      flags=re.IGNORECASE)

    about_store = bool(re.search(r"\b(?:esa tienda|esa sucursal|esa misma tienda|le fue|sus ventas|cuanto vendio|sus productos)\b", q))
    asks_products = bool(re.search(r"\b(?:productos|sku)\b", q))
    if about_store and not store:
        return None
    if asks_products and store:
        return f"¿Qué productos tuvieron más ventas en {store} en {month} de 2026?"
    if about_store and store and (SALES_RE.search(q) or "le fue" in q or "esa tienda" in q):
        return f"¿Cuánto vendió {store} en {month} de 2026?"
    return None



def resolve_policy_followup(question: str, previous: VerifiedTurn | None) -> str | None:
    """Expand a clearly dependent policy follow-up into a NEW retrieval query.

    Previous answers are not returned from cache; the RAG runs again with the
    omitted topic supplied. For unrelated or ambiguous questions, do nothing.
    """
    if previous is None or not previous.is_grounded or previous.kind != "grounded_policy":
        return None
    q = _plain(question).strip(" ¿?¡!.,;:")
    prior = _plain(previous.question)
    # Only a dependent utterance qualifies; a new self-contained question
    # must never inherit unrelated business policy.
    if not re.match(r"^(?:y si|y entonces|y que pasa si|entonces que pasa si|pero si|y en ese caso)\b", q):
        return None
    topics = ("devoluc", "devolver", "reembols", "merma", "caduc", "garant")
    if not any(term in prior for term in topics):
        return None
    if any(term in q for term in topics):
        return None
    combined = f"{previous.question.rstrip(' ?¿!.')}. {question.strip()}"
    return combined if len(combined) <= 1000 else None
