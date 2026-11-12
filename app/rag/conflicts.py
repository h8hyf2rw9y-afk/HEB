"""Heuristic detection of contradictory policy passages.

Approach (deterministic, explainable):
1. From each retrieved passage keep only the sentences related to the question.
2. Extract *claims*: quantities with units (días, horas, %, $...) and normative
   polarity (permitido / prohibido).
3. Two claims from different passages conflict when their sentences are about the
   same topic (token Jaccard similarity) but state different values for the same
   unit, or opposite polarity.
4. A conflict is only marked as resolved when a document *explicitly* says it
   replaces the other one or states a precedence rule. Effective dates are reported
   but never used on their own to pick a winner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from .index import SearchHit
from .models import ConflictClaim, DocumentInfo, PolicyConflict
from .text import normalize, split_sentences, tokenize

_NUMBER_WORDS = {
    "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
    "nueve": "9", "diez": "10", "doce": "12", "quince": "15", "veinte": "20", "treinta": "30",
    "cuarenta": "40", "sesenta": "60", "noventa": "90", "cien": "100",
}
_NUM = r"(\d+(?:[.,]\d+)?|" + "|".join(_NUMBER_WORDS) + r")"
_QUANTITY_RE = re.compile(
    _NUM + r"\s*(%|por ciento|dias? habiles|dias? naturales|dias? calendario|dias?|horas?|minutos?"
    r"|semanas?|meses|mes|anos?|pesos|dolares|usd|mxn)(?!\w)"
)
_CURRENCY_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_NEGATIVE_RE = re.compile(
    r"\b(no se permite|no esta permitido|no podra|no podran|no puede|no pueden|no debe|no deben"
    r"|prohibid[oa]s?|queda prohibido|esta prohibido|no se aceptan?|no aplica)\b"
)
_POSITIVE_RE = re.compile(
    r"\b(se permite|esta permitido|podra|podran|puede|pueden|se aceptan?|debe|deben|es obligatorio)\b"
)
_UNIT_TOKENS = set(tokenize("dia dias hora horas minuto minutos semana semanas mes meses ano anos "
                            "habiles naturales calendario pesos dolares por ciento"))


def _canonical_unit(unit: str) -> str:
    if unit.startswith("dia"):
        return "días hábiles" if "habil" in unit else "días"
    return {
        "%": "%", "por ciento": "%", "hora": "horas", "horas": "horas", "minuto": "minutos",
        "minutos": "minutos", "semana": "semanas", "semanas": "semanas", "mes": "meses",
        "meses": "meses", "ano": "años", "anos": "años", "pesos": "moneda", "dolares": "moneda",
        "usd": "moneda", "mxn": "moneda",
    }.get(unit, unit)


def _num(value: str) -> str:
    value = _NUMBER_WORDS.get(value, value).replace(",", ".")
    try:
        f = float(value)
    except ValueError:
        return value
    return str(int(f)) if f.is_integer() else str(f)


def extract_quantities(sentence: str) -> dict[str, set[str]]:
    norm = normalize(sentence)
    out: dict[str, set[str]] = {}
    for value, unit in _QUANTITY_RE.findall(norm):
        out.setdefault(_canonical_unit(unit), set()).add(_num(value))
    for value in _CURRENCY_RE.findall(norm):
        out.setdefault("moneda", set()).add(_num(value.replace(",", "")))
    return out


def polarity(sentence: str) -> int:
    norm = normalize(sentence)
    if _NEGATIVE_RE.search(norm):
        return -1
    if _POSITIVE_RE.search(norm):
        return 1
    return 0


def topic_tokens(sentence: str) -> set[str]:
    norm = _NEGATIVE_RE.sub(" ", normalize(sentence))
    return {t for t in tokenize(norm) if not t[:1].isdigit() and t not in _UNIT_TOKENS and t != "no"}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass
class _Sentence:
    hit: SearchHit
    text: str
    topic: set[str]
    quantities: dict[str, set[str]]
    polarity: int


def _relevant_sentences(question: str, hits: list[SearchHit]) -> list[_Sentence]:
    q_topic = topic_tokens(question)
    out: list[_Sentence] = []
    for hit in hits:
        for sent in split_sentences(hit.chunk.text):
            topic = topic_tokens(sent)
            if q_topic and not (topic & q_topic):
                continue
            out.append(_Sentence(hit, sent, topic, extract_quantities(sent), polarity(sent)))
    return out


def _same_location(a: SearchHit, b: SearchHit) -> bool:
    return a.chunk.chunk_id == b.chunk.chunk_id or (
        a.chunk.document == b.chunk.document
        and a.chunk.page == b.chunk.page
        and a.chunk.section == b.chunk.section
    )


def _mentions(phrase: str, info: DocumentInfo) -> bool:
    """True if ``phrase`` names the document by file name, file stem or title.

    Matching is token-boundary aware so ``..._v1`` does not match ``..._v10`` and
    "Política v1" does not match "Política v1.2".
    """
    p = normalize(phrase)
    stem = info.document.rsplit(".", 1)[0]
    candidates = {normalize(info.document), normalize(stem), normalize(stem.replace("_", " "))}
    if info.title:
        candidates.add(normalize(info.title))
    return any(
        len(c) >= 4  # very short stems ("a", "v1") collide with ordinary words
        and re.search(r"(?<![a-z0-9_])" + re.escape(c) + r"(?![a-z0-9_]|[.,][0-9])", p)
        for c in candidates
    )


def _unique_target(phrase: str, documents: dict[str, DocumentInfo]) -> DocumentInfo | None:
    """The single indexed document the phrase refers to, or None if none/ambiguous."""
    matched = [d for d in documents.values() if _mentions(phrase, d)]
    return matched[0] if len(matched) == 1 else None


def resolve_precedence(
    doc_a: str, doc_b: str, documents: dict[str, DocumentInfo]
) -> tuple[str | None, str]:
    """Return (preferred_document, basis).

    Only an explicit, unambiguous statement in the source resolves a conflict:
    - document X says it replaces Y, and the phrase identifies exactly one indexed
      document (Y); or
    - a precedence rule ("en caso de conflicto prevalece ...") that identifies
      exactly one of the two documents.
    Effective dates are reported but never decide on their own, and an ambiguous
    phrase (matching both versions or several documents) leaves the conflict
    unresolved.
    """
    a, b = documents.get(doc_a), documents.get(doc_b)
    if a is None or b is None or doc_a == doc_b:
        return None, "Contradicción dentro del mismo documento o sin metadatos; requiere revisión."

    ambiguous: list[str] = []
    for winner, loser in ((a, b), (b, a)):
        for phrase in winner.supersedes:
            if not _mentions(phrase, loser):
                continue
            if _unique_target(phrase, documents) is loser:
                return winner.document, f'{winner.document} indica: "{phrase}"'
            ambiguous.append(f'{winner.document}: "{phrase}"')
    for source in (a, b):
        for phrase in source.precedence_statements:
            target = _unique_target(phrase, documents)
            if target is a or target is b:
                return target.document, f'{source.document} indica: "{phrase}"'
            if _mentions(phrase, a) or _mentions(phrase, b):
                ambiguous.append(f'{source.document}: "{phrase}"')

    basis = "Ningún documento establece explícitamente y sin ambigüedad cuál prevalece."
    if ambiguous:
        basis += " Frases de reemplazo/precedencia ambiguas: " + "; ".join(ambiguous) + "."
    dates = [f"{d.document}: {d.effective_date}" for d in (a, b) if d.effective_date]
    if dates:
        basis += " Fechas de vigencia encontradas en el texto (no deciden por sí solas): " + "; ".join(dates) + "."
    return None, basis


def detect_conflicts(
    question: str,
    hits: list[SearchHit],
    documents: dict[str, DocumentInfo],
    *,
    numeric_similarity: float = 0.3,
    modality_similarity: float = 0.5,
) -> list[PolicyConflict]:
    sentences = _relevant_sentences(question, hits)
    conflicts: list[PolicyConflict] = []
    seen: set[tuple] = set()

    def claim(s: _Sentence, values: set[str] | None = None) -> ConflictClaim:
        return ConflictClaim(
            document=s.hit.chunk.document,
            page=s.hit.chunk.page,
            section=s.hit.chunk.section,
            sentence=s.text,
            values=sorted(values or []),
        )

    for s1, s2 in combinations(sentences, 2):
        if _same_location(s1.hit, s2.hit):
            continue
        sim = _jaccard(s1.topic, s2.topic)
        found: list[tuple[str, str, set[str], set[str]]] = []
        if sim >= numeric_similarity:
            for unit in s1.quantities.keys() & s2.quantities.keys():
                v1, v2 = s1.quantities[unit], s2.quantities[unit]
                if v1.isdisjoint(v2):
                    found.append(("numeric", unit, v1, v2))
        if sim >= modality_similarity and s1.polarity * s2.polarity == -1:
            found.append(("modality", "permitido/prohibido", set(), set()))

        for kind, topic, v1, v2 in found:
            key = (kind, topic, *sorted([s1.hit.chunk.chunk_id, s2.hit.chunk.chunk_id]))
            if key in seen:
                continue
            seen.add(key)
            preferred, basis = resolve_precedence(s1.hit.chunk.document, s2.hit.chunk.document, documents)
            conflicts.append(
                PolicyConflict(
                    kind=kind,  # type: ignore[arg-type]
                    topic=topic,
                    claims=[claim(s1, v1), claim(s2, v2)],
                    resolution="resolved_by_source" if preferred else "unresolved",
                    preferred_document=preferred,
                    basis=basis,
                )
            )
    return conflicts
