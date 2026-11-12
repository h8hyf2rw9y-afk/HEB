"""Evidence selection and extractive answer composition (no LLM).

Pipeline used by :class:`app.rag.service.PolicyRetriever` after retrieval:

1. ``analyse_question`` — content terms, soft alternatives, and the *focus* of
   "¿Cuál es el <X> de ...?" / "¿Cuánto cuesta ...?" questions.
2. ``assess_hit`` — how directly a passage answers: term coverage, overlap with
   its section heading (FAQ headings are the question itself) and, for table
   rows, whether the row label (entity) and a column header (attribute) match.
3. ``table_entity_gaps`` — a question about an entity a table does not list
   ("tolerancia de merma para electrónica") is detected, so the system abstains
   instead of returning rows for other entities.
4. ``compose_answer`` — direct answer from the best passage (all its relevant
   and restrictive sentences), only clearly useful supporting passages,
   contradictions, and the question terms the documents do not cover.

Every sentence in the answer is copied from a cited passage; nothing is
paraphrased or generated.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .conflicts import extract_quantities, topic_tokens
from .index import DEFINITIONAL_RE, SearchHit
from .models import Evidence, PolicyConflict
from .text import normalize, split_sentences, tokenize

PREFIX = 5
# Weight of the normalised retrieval score in the rerank (0 = coverage-first).
# 0.0, 0.3, 0.5, 0.7, 0.85 and 1.0 were compared on the development set only.
RETRIEVAL_W = 0.0
HEADING_W = 0.5  # 0.3 and 0.5 compared on the development set
SAME_QUESTION = 0.8  # heading overlap at which an FAQ entry is the question itself

# "¿Cuál es el precio de ...?", "¿Cuánto cuesta ...?", "¿Qué sueldo ...?"
_FOCUS_PATTERNS = [
    re.compile(r"^\W*(?:cual|que)\s+(?:es|son)\s+(?:el|la|los|las)\s+([a-z]+)"),
    re.compile(r"\bcuanto\s+(cuesta|cuestan|cobra|cobran|gana|ganan|se\s+paga)\b"),
    re.compile(r"\bque\s+(precio|costo|sueldo|salario|marca|color|contrasena)\b"),
]
# Sentences that carry conditions or prohibitions are kept with the direct answer.
_RESTRICTION_RE = re.compile(
    r"\b(no|nunca|solo|solamente|unicamente|exclusivamente|en ningun caso|bajo ninguna|"
    r"prohibid[oa]|requiere|requieren|siempre que|salvo|excepto|como maximo|como minimo|"
    r"al menos|hasta|sin)\b"
)
_CROSS_REFERENCE_RE = re.compile(r"^\W*(consulta|ver|vease)\b")


def _pfx(tokens: set[str]) -> set[str]:
    return {t[:PREFIX] for t in tokens}


@dataclass
class Question:
    text: str
    terms: set[str]  # topic tokens (stemmed, no stopwords, no units/numbers)
    focus: set[str] = field(default_factory=set)
    definitional: bool = False  # "¿Qué es ...?"
    quantities: dict[str, set[str]] = field(default_factory=dict)  # "3 días" -> {"días": {"3"}}


def analyse_question(question: str) -> Question:
    norm = normalize(question)
    focus: set[str] = set()
    for pattern in _FOCUS_PATTERNS:
        m = pattern.search(norm)
        if m:
            focus = set(tokenize(m.group(1)))
            break
    return Question(
        text=question, terms=topic_tokens(question), focus=focus,
        definitional=bool(DEFINITIONAL_RE.search(question)),
        quantities=extract_quantities(question),
    )


# ------------------------------------------------------------------ passages
def row_cells(text: str) -> list[tuple[str, str]]:
    """("Header", "value") pairs of a table-row passage "H: v | H: v."."""
    cells = []
    for part in text.rstrip(".").split(" | "):
        head, sep, value = part.partition(": ")
        cells.append((head, value) if sep else ("", part))
    return cells


@dataclass
class HitAssessment:
    hit: SearchHit
    heading_match: float  # share of question terms present in the section heading
    entity_match: bool  # table row: label matches a question term
    attribute_match: bool  # table row: a non-label header matches a question term
    rank_score: float = 0.0


def _matches(tokens: set[str], q_pfx: set[str]) -> bool:
    return bool(_pfx(tokens) & q_pfx)


def assess_hit(q: Question, hit: SearchHit, q_pfx: set[str], max_score: float = 0.0) -> HitAssessment:
    chunk = hit.chunk
    # Heading overlap is measured on the question's own terms (not expansions).
    own = _pfx(q.terms)
    heading = topic_tokens(chunk.section or "")
    heading_match = len(_pfx(heading) & own) / len(own) if own else 0.0
    entity = attribute = strong = False
    if chunk.kind == "table_row":
        from .conflicts import _label_tokens  # negation-aware label tokens

        label = _pfx(_label_tokens(chunk.label or ""))
        entity = bool(label & q_pfx)
        strong = bool(label) and len(label & q_pfx) / len(label) >= 0.6
        if not entity and q.quantities:  # "3 días" in the question vs label "3 días"
            label_q = extract_quantities(chunk.label or "")
            entity = strong = any(label_q.get(u, set()) & v for u, v in q.quantities.items())
        cells = row_cells(chunk.text)
        headers = [h for h, _ in cells[1:]]
        attribute = any(_matches(topic_tokens(h), q_pfx) for h in headers)
        # The asked entity may sit in another column ("Quién autoriza: Gerente de
        # Tienda" while the question asks for the amount in "Rango de monto").
        for i, (head, value) in enumerate(cells[1:], start=1):
            vt = _pfx(topic_tokens(value))
            others = [h for j, (h, _) in enumerate(cells) if j != i]
            if len(vt) >= 2 and len(vt & q_pfx) / len(vt) >= 0.6 and any(
                _matches(topic_tokens(h), q_pfx) for h in others
            ):
                entity = strong = attribute = True
                break
    a = HitAssessment(hit, heading_match, entity, attribute)
    # How directly the passage answers: retrieval score (normalised by the best in
    # the pool), term coverage, heading and entity matches.
    rel = hit.score / max_score if max_score > 0 else 0.0
    a.rank_score = (
        RETRIEVAL_W * rel + (1 - RETRIEVAL_W) * hit.coverage
        + HEADING_W * heading_match + 0.3 * (entity and (strong or attribute)) + 0.2 * (strong and attribute)
    )
    if q.definitional and "definicion" in normalize(chunk.section or ""):
        a.rank_score += 0.5
    return a


def rerank(assessments: list[HitAssessment]) -> list[HitAssessment]:
    return sorted(assessments, key=lambda a: (-a.rank_score, -a.hit.score))


def prune_table_siblings(assessments: list[HitAssessment]) -> list[HitAssessment]:
    """When a table has a row for the asked entity, its other rows answer for
    other entities ("5 días", "4 días" when "3 días" was asked) and are dropped."""
    matched_tables = {
        (a.hit.chunk.document, a.hit.chunk.section)
        for a in assessments if a.hit.chunk.kind == "table_row" and a.entity_match
    }
    return [
        a for a in assessments
        if a.hit.chunk.kind != "table_row" or a.entity_match
        or (a.hit.chunk.document, a.hit.chunk.section) not in matched_tables
    ]


@dataclass
class TableGap:
    document: str
    page: int
    section: str | None
    missing: set[str]
    labels: list[str]


def table_entity_gaps(q: Question, assessments: list[HitAssessment], weight: Callable[[str], float]) -> list[TableGap]:
    """Tables that match the question's attribute but list no row for the asked
    entity: the question names a term found in none of the rows nor the heading."""
    groups: dict[tuple[str, str | None], list[HitAssessment]] = {}
    for a in assessments:
        if a.hit.chunk.kind == "table_row":
            groups.setdefault((a.hit.chunk.document, a.hit.chunk.section), []).append(a)
    gaps = []
    for (doc, section), rows in groups.items():
        # Only tables whose *column headers* match the question (the attribute
        # asked, e.g. "Tolerancia", "Plazo"); a heading match alone is not enough.
        if any(r.entity_match for r in rows) or not any(r.attribute_match for r in rows):
            continue
        present = set().union(*(_pfx(topic_tokens(r.hit.chunk.text)) for r in rows)) | _pfx(topic_tokens(section or ""))
        missing = {t for t in q.terms if t[:PREFIX] not in present and weight(t) > 0}
        if missing:
            gaps.append(TableGap(doc, rows[0].hit.chunk.page, section, missing,
                                 [r.hit.chunk.label or "" for r in rows]))
    return gaps


# ----------------------------------------------------------------- sentences
def _sentences(passage: str) -> list[str]:
    """Sentences, with very short ones ("No.", "Sí, con restricciones.") merged forward."""
    out: list[str] = []
    carry = ""
    for s in split_sentences(passage) or [passage]:
        if len(s.split()) < 4:
            carry = f"{carry} {s}".strip()
            continue
        out.append(f"{carry} {s}".strip() if carry else s)
        carry = ""
    if carry:
        out.append(carry)
    return out


def _score(sentence: str, q_pfx: set[str], weight: Callable[[str], float]) -> float:
    return sum(weight(t) for t in sorted(topic_tokens(sentence)) if t[:PREFIX] in q_pfx)


def excerpt(q: Question, a: HitAssessment, weight: Callable[[str], float], *, primary: bool, limit: int = 5) -> str:
    """Sentences of a passage that answer the question, in document order.

    Table rows are quoted whole. Otherwise a sentence is kept when it scores at
    least 60% of the best sentence (IDF-weighted question terms) and shares at
    least half as many question terms as the best sentence, or, in the primary
    passage, states a restriction ("solo",
    "no", "exclusivamente", ...) with at least 40% of the best score. The first
    sentence of an FAQ entry whose heading is the question is kept. If no kept
    sentence contains the question focus, the best one that does is added. A kept
    sentence that only matches the focus ("Tiene vigencia de doce meses.")
    brings its predecessor, which names the subject.
    """
    e = a.hit.chunk
    if e.kind != "text":
        return e.text
    q_pfx = _pfx(q.terms)
    focus_pfx = _pfx(q.focus)
    sents = _sentences(e.text)
    if not sents:
        return e.text
    scores = [_score(s, q_pfx, weight) for s in sents]
    for i, s in enumerate(sents):  # "Vale de tienda: documento ..." defines the asked term
        head, sep, _ = s.partition(": ")
        if sep and len(head.split()) <= 6 and _pfx(topic_tokens(head)) and _pfx(topic_tokens(head)) <= q_pfx:
            scores[i] *= 2
    best = max(scores)
    counts = [len(_pfx(topic_tokens(s)) & q_pfx) for s in sents]
    best_count = counts[scores.index(best)] if best > 0 else 0
    chosen: set[int] = set()
    for i, (s, sc) in enumerate(zip(sents, scores)):
        norm = normalize(s)
        if _CROSS_REFERENCE_RE.match(norm):
            continue
        toks = _pfx(topic_tokens(s))
        if sc > 0 and sc >= 0.6 * best and counts[i] >= 0.5 * best_count:
            chosen.add(i)
        elif (
            primary and sc >= 0.4 * best and _RESTRICTION_RE.search(norm)
            and len(toks & q_pfx) >= min(2, len(q_pfx))
        ):
            chosen.add(i)
        elif primary and a.heading_match >= SAME_QUESTION:
            # The heading *is* the question (FAQ): every sentence answers it.
            chosen.add(i)
        elif primary and i == 0 and a.heading_match >= 0.6:
            # A closely matching FAQ heading: its first sentence is the direct answer.
            chosen.add(i)
    # The focus ("vigencia" in "¿Cuál es la vigencia de ...?") must be in the excerpt.
    if focus_pfx and not any(_pfx(topic_tokens(sents[i])) & focus_pfx for i in chosen):
        with_focus = [i for i, s in enumerate(sents) if _pfx(topic_tokens(s)) & focus_pfx]
        if with_focus:
            chosen.add(max(with_focus, key=lambda i: scores[i]))
    for i in sorted(chosen):
        if i > 0 and i - 1 not in chosen and not ((_pfx(topic_tokens(sents[i])) - focus_pfx) & q_pfx):
            chosen.add(i - 1)
    if not chosen:
        chosen = {max(range(len(sents)), key=lambda i: scores[i])}
    return " ".join(sents[i] for i in sorted(chosen)[:limit])


# ------------------------------------------------------------------- answer
def cite(e: Evidence) -> str:
    ref = f"{e.document}, p. {e.page}"
    return f"[{ref}, sección: {e.section}]" if e.section else f"[{ref}]"


_CLITIC = re.compile(r"(?:le|les|la|las|lo|los|se|me|nos)$")


def unknown_terms(q: Question, known: Callable[[str], bool], known_prefix: Callable[[str], bool]) -> list[str]:
    """Question terms that appear nowhere in the indexed documents, not even as
    another inflection (same 4-letter root) or as a verb with a clitic ("darle")."""
    out = []
    for t in sorted(q.terms):
        if known(t) or known_prefix(t[:4]):
            continue
        bare = _CLITIC.sub("", t) if len(t) >= 5 else t
        if len(bare) < 4 or known_prefix(bare[:4]):
            continue
        out.append(t)
    return out


def compose_answer(
    q: Question,
    ranked: list[HitAssessment],
    evidence_by_id: dict[str, Evidence],
    conflicts: list[PolicyConflict],
    weight: Callable[[str], float],
    *,
    known: Callable[[str], bool] | None = None,
    known_prefix: Callable[[str], bool] | None = None,
    normative_docs: set[str] | None = None,
    subordinate_docs: set[str] | None = None,
    original_words: dict[str, str] | None = None,
    max_supporting: int = 2,
) -> str:
    """Three blocks: what the documents confirm, contradictions, and unknown terms.

    - The primary passage is the best-ranked one; a document that the sources
      say does not prevail never leads, and when an FAQ (a document that defers
      to policies/procedures) leads and a policy/procedure passage ranks within
      90% of it, the policy/procedure leads (the FAQ can still support it).
    - A supporting passage is added only if it is part of a detected conflict,
      or it answers almost as well (rank >= 85% and coverage >= 90% of the
      primary), contains every question term the primary contains, and either is
      a near tie (>= 95%) in the same section (e.g. the checklist row next to the
      section text) or corroborates the primary from another document (Jaccard
      >= 0.2 on words beyond the question terms).
      Definitional questions only take passages from "Definiciones" sections.
    """
    normative_docs = normative_docs or set()
    subordinate_docs = subordinate_docs or set()
    superseded = {
        cl.document for c in conflicts if c.preferred_document for cl in c.claims if cl.document != c.preferred_document
    }
    conflict_ids = {(cl.document, cl.page) for c in conflicts for cl in c.claims}
    primary = ranked[0]
    if primary.hit.chunk.document in superseded:
        primary = next((a for a in ranked if a.hit.chunk.document not in superseded), primary)
    if primary.hit.chunk.document in subordinate_docs:
        for a in ranked:
            if a.hit.chunk.document in normative_docs and a.rank_score >= 0.9 * primary.rank_score:
                primary = a
                break

    def is_definition(a: HitAssessment) -> bool:
        return "definicion" in normalize(a.hit.chunk.section or "")

    primary_terms = _pfx(topic_tokens(primary.hit.chunk.text))
    q_pfx_all = _pfx(q.terms)

    def terms_in(a: HitAssessment) -> set[str]:
        c = a.hit.chunk
        return q_pfx_all & _pfx(topic_tokens(f"{c.section or ''} {c.text}"))
    chosen = [primary]
    for a in ranked:
        if a is primary or len(chosen) > max_supporting:
            continue
        c = a.hit.chunk
        if (c.document, c.page) in conflict_ids:
            chosen.append(a)
            continue
        if q.definitional and not is_definition(a):
            continue
        close = a.rank_score >= 0.85 * primary.rank_score and a.hit.coverage >= 0.9 * primary.hit.coverage
        near_tie = a.rank_score >= 0.95 * primary.rank_score
        # Corroboration: another document stating the same content, i.e. sharing
        # words beyond the question terms with the primary passage.
        own = _pfx(topic_tokens(c.text)) - q_pfx_all
        base = primary_terms - q_pfx_all
        corroborates = c.document != primary.hit.chunk.document and bool(own and base) and (
            len(own & base) / len(own | base) >= 0.2
        )
        same_section = c.document == primary.hit.chunk.document and c.section == primary.hit.chunk.section
        # It must also contain every question term the primary passage contains.
        if close and terms_in(primary) <= terms_in(a) and (near_tie and same_section or corroborates):
            chosen.append(a)

    lines = ["Respuesta según los documentos:"]
    quoted: list[str] = []
    for i, a in enumerate(chosen):
        e = evidence_by_id[a.hit.chunk.chunk_id]
        text = excerpt(q, a, weight, primary=(i == 0))
        if text in quoted:
            continue
        quoted.append(text)
        note = " (no prevalece según la fuente; ver contradicción)" if e.document in superseded else ""
        lines.append(f"- {text} {cite(e)}{note}")

    for c in conflicts:
        sides = "; ".join(f"{cl.document} p. {cl.page}: {', '.join(cl.values) or cl.sentence}" for cl in c.claims)
        if c.resolution == "resolved_by_source":
            lines.append(f"⚠️ Contradicción entre documentos ({c.topic}): {sides}. Prevalece {c.preferred_document}. {c.basis}")
        else:
            lines.append(
                f"⚠️ Política contradictoria sin resolver ({c.topic}): {sides}. {c.basis} "
                "Se recomienda confirmar con el área responsable."
            )

    missing = unknown_terms(q, known, known_prefix or (lambda _p: False)) if known else []
    if missing:
        words = [(original_words or {}).get(t, t) for t in missing]
        lines.append("ℹ️ Los documentos no mencionan: " + ", ".join(words) + "; la respuesta cubre solo lo demás.")
    return "\n".join(lines)


def surface_words(question: str) -> dict[str, str]:
    """Map stemmed tokens back to the words the user typed (for messages)."""
    out: dict[str, str] = {}
    for word in re.findall(r"[\wáéíóúñü]+", question.lower()):
        for t in tokenize(word):
            out.setdefault(t, word)
    return out
