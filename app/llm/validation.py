"""Conservative checks on a model rewrite of a verified answer.

The checks are deliberately strict: when in doubt, the rewrite is rejected and
the deterministic answer is used. They catch altered or invented figures,
dates, SKUs, citations and sources, dropped contradiction warnings, chosen
winners for unresolved conflicts, removed negations, and swapped
entity → figure pairs. They do NOT guarantee the absence of hallucinations: a
rewrite can still add a qualitative claim without numbers or citations, or
change meaning without touching any protected token (see app/llm/README.md).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

CITATION_RE = re.compile(r"\[[^\[\]]*?\.pdf, p\. \d+[^\[\]]*\]")
FILE_RE = re.compile(r"\b[\w-]+\.(?:pdf|csv)\b", re.IGNORECASE)
PAGE_REF_RE = re.compile(r"\b[\w-]+\.pdf p\. \d+", re.IGNORECASE)  # "faq.pdf p. 4" in conflict notes
DOC_CODE_RE = re.compile(r"\b[A-Z]{2,4}-[A-Z]{2,4}-\d{3}\b")
SKU_RE = re.compile(r"\bSKU-\d+\b", re.IGNORECASE)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.,])\d+(?:\.\d+)?(?![\w])")
LIST_MARKER_RE = re.compile(r"^\s*\d{1,2}[.)]\s+", re.MULTILINE)
MAX_RESULTS_RE = re.compile(r"\(m[aá]ximo \d+ resultados\)", re.IGNORECASE)
MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "setiembre", "octubre", "noviembre", "diciembre")
NEGATION_RE = re.compile(r"\b(no|nunca|jamas|ningun[oa]?|prohibid[oa]s?|tampoco)\b")
RESTRICTION_RE = re.compile(r"\b(solo|solamente|unicamente|exclusivamente)\b")
ABSTENTION_RE = re.compile(r"\bno (encontre|tengo|cuento con|dispongo de|hay)\b.{0,40}\binformacion\b")
SQL_RE = re.compile(r"\bselect\b[\s\S]{0,200}\bfrom\b|read_csv|\bduckdb\.connect\b", re.IGNORECASE)
INJECTION_RE = re.compile(
    r"\b(ignora|ignore|olvida|disregard)\b.{0,60}\b(instrucci\w*|instructions?|reglas|anteriores|previous)\b"
    r"|\b(prompt del sistema|system prompt|mensaje del sistema)\b|\beres ahora\b|\bact as\b",
    re.IGNORECASE,
)

_UNITS = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
          "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
          "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
          "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
          "veintisiete": 27, "veintiocho": 28, "veintinueve": 29}
_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
_HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300, "cuatrocientos": 400,
             "quinientos": 500, "seiscientos": 600, "setecientos": 700, "ochocientos": 800, "novecientos": 900}
_SIMPLE = {"uno": 1, "un": 1, "una": 1}


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _canonical(number: str) -> str:
    try:
        return "n:" + format(Decimal(number.replace(",", "")).normalize(), "f")
    except InvalidOperation:
        return "n:" + number


def _word_numbers(plain: str) -> list[str]:
    """Spanish number words → canonical values ("cuarenta y ocho" → 48, "tres mil" → 3000).
    Bare articles ("un", "una") only count when followed by "mil"."""
    tokens = re.findall(r"[a-z]+", plain)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        value, consumed = 0, 0
        j = i
        while j < len(tokens):
            t = tokens[j]
            if t in _HUNDREDS and value % 1000 == 0:
                value += _HUNDREDS[t]
            elif t in _TENS and value % 100 == 0:
                value += _TENS[t]
                if j + 2 < len(tokens) and tokens[j + 1] == "y" and tokens[j + 2] in _UNITS and _UNITS[tokens[j + 2]] < 10:
                    value += _UNITS[tokens[j + 2]]
                    j += 2
            elif t in _UNITS and value % 100 == 0:
                value += _UNITS[t]
            elif t in _SIMPLE and j + 1 < len(tokens) and tokens[j + 1] == "mil" and value == 0:
                value = 1
            elif t == "mil" and value > 0:
                value *= 1000
            else:
                break
            j += 1
            consumed = j - i
        if consumed:
            out.append(f"n:{value}")
            i += consumed
        else:
            i += 1
    return out


@dataclass
class Facts:
    values: set[str] = field(default_factory=set)  # numbers, times, dates, SKUs, months
    citations: set[str] = field(default_factory=set)
    files: set[str] = field(default_factory=set)
    codes: set[str] = field(default_factory=set)


def extract_facts(text: str, *, drop_optional: bool = False) -> Facts:
    facts = Facts()
    facts.citations = set(CITATION_RE.findall(text))
    body = CITATION_RE.sub(" ", text)
    facts.files = {f.lower() for f in FILE_RE.findall(text)}
    body = PAGE_REF_RE.sub(" ", body)
    facts.codes = set(DOC_CODE_RE.findall(body))
    body = DOC_CODE_RE.sub(" ", body)
    if drop_optional:
        body = MAX_RESULTS_RE.sub(" ", body)
    body = LIST_MARKER_RE.sub(" ", body)
    body = FILE_RE.sub(" ", body)
    for sku in SKU_RE.findall(body):
        facts.values.add("sku:" + sku.upper())
    body = SKU_RE.sub(" ", body)
    for d in ISO_DATE_RE.findall(body):
        facts.values.add("d:" + d)
    body = ISO_DATE_RE.sub(" ", body)
    for t in TIME_RE.findall(body):
        h, m = t.split(":")
        facts.values.add(f"t:{int(h)}:{m}")
    body = TIME_RE.sub(" ", body)
    facts.values.update(_canonical(n) for n in NUMBER_RE.findall(body))
    plain = _plain(body)
    facts.values.update(_word_numbers(plain))
    facts.values.update("m:" + ("septiembre" if m == "setiembre" else m) for m in MONTHS if re.search(rf"\b{m}\b", plain))
    return facts


def _line_keys(verified: str) -> list[tuple[str, set[str]]]:
    """(entity key, values) for verified lines shaped "Entidad: cifras" (data answers)."""
    pairs = []
    for line in verified.splitlines():
        line = LIST_MARKER_RE.sub("", line).strip()
        key, sep, rest = line.partition(":")
        if not sep or not rest.strip() or "(" in key or len(key) > 80:
            continue
        values = {v for v in extract_facts(rest).values if not v.startswith("m:")}
        if not values:
            continue
        sku = SKU_RE.search(key)
        pairs.append(((sku.group(0) if sku else key.strip()), values))
    return pairs


def _segments_by_citation(text: str) -> dict[str, str]:
    """Text preceding each citation (since the previous citation)."""
    out, last = {}, 0
    for m in CITATION_RE.finditer(text):
        out.setdefault(m.group(0), text[last:m.start()])
        last = m.end()
    return out


@dataclass
class ValidationResult:
    ok: bool
    failures: list[str]

    @property
    def reason(self) -> str:
        return "; ".join(self.failures)


def validate_rewrite(
    *,
    kind: str,
    question: str,
    verified: str,
    candidate: str,
    allowed_sources: set[str],
    conflicts: list[dict] | None = None,
) -> ValidationResult:
    """``kind``: "datos" or "politicas". ``allowed_sources``: file names in the evidence."""
    failures: list[str] = []
    cand = candidate.strip()
    if not cand:
        return ValidationResult(False, ["respuesta vacía"])
    if len(cand) > max(3 * len(verified), len(verified) + 600):
        failures.append("respuesta demasiado larga")

    v_all = extract_facts(verified)
    v_req = extract_facts(verified, drop_optional=True)
    q = extract_facts(question)
    c = extract_facts(cand)

    missing = sorted(v_req.values - c.values)
    if missing:
        failures.append("cifras o fechas omitidas o alteradas: " + ", ".join(missing))
    extra = set(c.values) - v_all.values - q.values
    # A rewrite may state how many items it lists ("hay tres productos") when that
    # count is written in words and equals the number of verified data rows.
    # Digits are never exempted, so "3 días" stays a new figure.
    row_count = len(_line_keys(verified)) if kind == "datos" else 0
    if 2 <= row_count <= 20:
        spelled = {v for v in extra if v == f"n:{row_count}" and not re.search(rf"(?<![\d.,]){row_count}(?![\d.,:])", CITATION_RE.sub(" ", cand))}
        extra -= spelled
    if extra:
        failures.append("cifras o fechas nuevas: " + ", ".join(sorted(extra)))

    # Store catalogue IDs are not ordinary prose: when a model summarizes
    # four stores, the exact T01/T02 identifiers must remain visible.
    if re.search(r"(?m)^\s*-\s*T\d{2}:", verified):
        store_codes = set(re.findall(r"\bT\d{2}\b", verified))
        output_codes = set(re.findall(r"\bT\d{2}\b", cand))
        if store_codes - output_codes:
            failures.append("códigos de tienda omitidos: " + ", ".join(sorted(store_codes - output_codes)))
        if output_codes - store_codes:
            failures.append("códigos de tienda desconocidos: " + ", ".join(sorted(output_codes - store_codes)))

    invented = sorted(c.citations - v_all.citations)
    if invented:
        failures.append("citas inexistentes: " + " ".join(invented))
    dropped = sorted(v_all.citations - c.citations)
    if dropped:
        failures.append("citas omitidas: " + " ".join(dropped))
    unknown_files = sorted(c.files - v_all.files - {s.lower() for s in allowed_sources})
    if unknown_files:
        failures.append("fuentes inexistentes: " + ", ".join(unknown_files))
    new_codes = sorted(c.codes - v_all.codes)
    if new_codes:
        failures.append("códigos de documento nuevos: " + ", ".join(new_codes))

    # Each citation must keep the figures it supports before it.
    if kind == "politicas":
        v_seg, c_seg = _segments_by_citation(verified), _segments_by_citation(cand)
        for cit, seg in v_seg.items():
            if cit not in c_seg:
                continue
            need = {v for v in extract_facts(seg, drop_optional=True).values if not v.startswith("m:")}
            have = extract_facts(c_seg[cit]).values
            if need - have:
                failures.append(f"cifras separadas de su cita {cit}")
        # Every substantial paragraph must carry a citation (a citation at the end of a
        # paragraph covers the sentences before it). Conflict notes are exempt.
        for block in re.split(r"\n+", cand):
            if (len(block.split()) >= 18 and not CITATION_RE.search(block)
                    and not re.search(r"contradic|prevalece|sin resolver", _plain(block))):
                failures.append("párrafo largo sin cita")
                break

    # Data answers: each "Entidad: cifras" pair must stay together (no swapped figures).
    if kind == "datos":
        keys = _line_keys(verified)
        positions = []
        for key, _ in keys:
            idx = cand.find(key)
            if idx >= 0:
                positions.append(idx)
        for key, values in keys:
            idx = cand.find(key)
            if idx < 0:
                failures.append(f"falta «{key}»")
                continue
            nxt = min([p for p in positions if p > idx] + [len(cand)])
            window = extract_facts(cand[idx:nxt]).values
            if values - window:
                failures.append(f"cifras de «{key}» alteradas o reasignadas")

    p_cand, p_ver = _plain(cand), _plain(verified)
    if conflicts:
        if "contradic" not in p_cand:
            failures.append("se omitió la advertencia de contradicción")

        def prevails(doc: str) -> bool:  # "prevalece X" but not "no prevalece X"
            return bool(re.search(r"(?<!no )(?<!no se )prevalece\s+(?:el\s+|la\s+)?" + re.escape(doc), cand, re.IGNORECASE))

        for conflict in conflicts:
            preferred = conflict.get("preferred_document")
            docs = [cl.get("document", "") for cl in conflict.get("claims", []) if cl.get("document")]
            if conflict.get("resolution") == "resolved_by_source" and preferred:
                if not prevails(preferred):
                    failures.append(f"se omitió que prevalece {preferred} según la fuente")
                for doc in docs:
                    if doc != preferred and prevails(doc):
                        failures.append(f"declara que prevalece {doc}, contra la fuente")
            else:
                if "sin resolver" not in p_cand:
                    failures.append("no indica que la contradicción está sin resolver")
                if re.search(r"(?<!no )prevalece\s+(?:el\s+|la\s+)?[\w-]+\.pdf", cand, re.IGNORECASE):
                    failures.append("elige un documento ganador en una contradicción sin resolver")

    if len(NEGATION_RE.findall(p_cand)) < len(NEGATION_RE.findall(p_ver)):
        failures.append("se perdieron negaciones")
    if len(RESTRICTION_RE.findall(p_cand)) < len(RESTRICTION_RE.findall(p_ver)):
        failures.append("se perdieron restricciones (solo/únicamente/exclusivamente)")
    if ABSTENTION_RE.search(p_cand) and not ABSTENTION_RE.search(p_ver):
        failures.append("declara falta de información en una respuesta verificada")
    if SQL_RE.search(cand):
        failures.append("contiene SQL o código")
    if INJECTION_RE.search(cand):
        failures.append("reproduce instrucciones inyectadas")
    return ValidationResult(not failures, failures)


def has_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))



@dataclass(frozen=True)
class ComposedRewrite:
    """Final checked text; may retain an immutable evidence appendix."""
    ok: bool
    answer: str
    failures: list[str]
    evidence_appended: bool = False


def compose_grounded_answer(*, kind: str, question: str, verified: str,
                            candidate: str, allowed_sources: set[str],
                            conflicts: list[dict] | None = None) -> ComposedRewrite:
    """Accept model-authored prose while preserving every verified CSV row.

    For exact policy rules/conflicts we do not loosen validation. For CSV
    summaries, a short model explanation may omit lower-ranked stores.
    Rather than rejecting it outright or trusting a partial table, attach
    the original complete verified rows and validate the ENTIRE composite.
    Any invented number, SKU, source, citation, unsafe text or contradiction
    still fails. This does not replace semantic evaluation of prose.
    """
    strict = validate_rewrite(kind=kind, question=question, verified=verified,
                              candidate=candidate, allowed_sources=allowed_sources,
                              conflicts=conflicts)
    if strict.ok:
        return ComposedRewrite(True, candidate.strip(), [])
    # Appending facts is only appropriate for CSV outputs without policy conflict.
    # A model that invented facts or attempted instructions must not be shown.
    if kind != "datos" or conflicts or not candidate.strip() or has_injection(candidate):
        return ComposedRewrite(False, verified, strict.failures)
    # Missing individual SKU, per-item existence, return constraints or
    # support-ticket details must never be disguised with an appendix:
    # those are operational instructions, not a simple store summary.
    summary = _plain(verified)
    safe_summary = (
        "ventas netas por tienda" in summary
        or "productos distintos por tienda" in summary
        or "tiendas registradas" in summary
    )
    if (not safe_summary or SKU_RE.search(verified)
            or "existencia" in summary or "punto de reorden" in summary):
        return ComposedRewrite(False, verified, strict.failures)
    if not all(
        issue.startswith(("cifras o fechas omitidas", "falta «", "cifras de «",
                          "códigos de tienda omitidos"))
        for issue in strict.failures
    ):
        return ComposedRewrite(False, verified, strict.failures)
    composite = candidate.strip() + "\n\n**Detalle verificado de los CSV:**\n" + verified
    checked = validate_rewrite(kind=kind, question=question, verified=verified,
                               candidate=composite, allowed_sources=allowed_sources,
                               conflicts=conflicts)
    if not checked.ok:
        return ComposedRewrite(False, verified, checked.failures)
    return ComposedRewrite(True, composite, [], evidence_appended=True)


# --- Data summaries: the model writes a short summary, the app shows the table ---

CAUSAL_RE = re.compile(
    r"\b(?:debido a|gracias a|se debe a|a causa de|porque|ya que|puesto que|esto se explica por|"
    r"por (?:sus|las|la|el|los|una|un) (?:promocion\w*|ubicacion\w*|campana\w*|marketing|publicidad|"
    r"trafico|afluencia|clientes|precios? mas bajos?))\b"
)
APPROX_RE = re.compile(r"\$?\s?(\d+(?:[.,]\d+)?)\s*(millones|mdp|mil)\b", re.IGNORECASE)
SUPERLATIVE_RE = re.compile(r"\b(?:mas vend\w*|vendio mas|mas ventas|mayores ventas|mayor venta|primer lugar|primera|lider\w*|la que mas)\b")
_MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


def _entity_keys(verified: str) -> list[tuple[str, set[str], int | None]]:
    """(entity, its figures, rank) for verified lines shaped "[1.|-] Entidad: cifras"."""
    out = []
    for line in verified.splitlines():
        rank_match = re.match(r"^\s*(\d{1,2})[.)]\s+", line)
        body = LIST_MARKER_RE.sub("", line).strip().lstrip("-•* ").strip()
        key, sep, rest = body.partition(":")
        if not sep or not rest.strip() or "(" in key or len(key) > 80 or len(key.split()) > 12:
            continue
        values = {v for v in extract_facts(rest).values if not v.startswith("m:")}
        if values:
            out.append((key.strip(), values, int(rank_match.group(1)) if rank_match else None))
    return out


def _aliases(key: str, all_keys: list[str] | None = None) -> list[str]:
    """Ways a summary can name an entity: the full key, the store without
    "Tienda", and for composite keys ("Tienda X · SKU-1013 · Bolillo pieza")
    each part that belongs to this row only (a shared store name does not)."""
    plain = _plain(key)
    names = [plain]
    if plain.startswith("tienda "):
        names.append(plain.removeprefix("tienda "))
    parts = [p.strip() for p in plain.split("·") if p.strip()]
    if len(parts) > 1:
        others = [_plain(k) for k in (all_keys or []) if k != key]
        names += [p for p in parts if not any(p in o for o in others)]
    return [n for n in names if len(n) >= 3]


def _resolve_approximations(text: str, verified: str) -> tuple[str, list[str]]:
    """Replace «2.16 millones» by the unique verified amount it rounds; an
    approximation that matches no amount, or more than one, is reported."""
    amounts = [float(m.replace(",", "")) for m in _MONEY_RE.findall(verified)]
    problems: list[str] = []

    def swap(match: re.Match) -> str:
        raw, unit = match.group(1), match.group(2).lower()
        number = float(raw.replace(",", "."))
        scale = 1_000 if unit == "mil" else 1_000_000
        decimals = len(re.split(r"[.,]", raw)[1]) if re.search(r"[.,]", raw) else 0
        hits = sorted({a for a in amounts if round(a / scale, decimals) == round(number, decimals)})
        if len(hits) == 1:
            return f"${hits[0]:,.2f}"
        problems.append(match.group(0))
        return match.group(0)

    return APPROX_RE.sub(swap, text), problems


def validate_data_summary(*, question: str, verified: str, summary: str,
                          allowed_sources: set[str]) -> ValidationResult:
    """A short model summary of a verified CSV result. It may omit rows (the full
    verified table is shown with it) but it may not add, alter, reassign or
    re-rank any figure, invent sources, or claim causes the data do not record."""
    failures: list[str] = []
    cand = summary.strip()
    if not cand:
        return ValidationResult(False, ["respuesta vacía"])
    if len(cand) > 1200:
        failures.append("resumen demasiado largo")
    resolved, approx = _resolve_approximations(cand, verified)
    if approx:
        failures.append("cifra aproximada sin correspondencia única: " + ", ".join(approx))
    v, q, c = extract_facts(verified), extract_facts(question), extract_facts(resolved)
    extra = set(c.values) - v.values - q.values
    keys = _entity_keys(verified)
    key_names = [k for k, _, _ in keys]
    if 2 <= len(keys) <= 20:  # "las cuatro tiendas" may name the row count in words
        extra -= {f"n:{len(keys)}"}
    if extra:
        failures.append("cifras o fechas nuevas: " + ", ".join(sorted(extra)))
    if c.citations:
        failures.append("citas en una respuesta de datos")
    unknown_files = sorted(c.files - v.files - {s.lower() for s in allowed_sources})
    if unknown_files:
        failures.append("fuentes inexistentes: " + ", ".join(unknown_files))

    # Each figure stays with its entity: within a sentence, figures belong to the
    # entity mentioned just before them (or the first one, if they come first).
    owners: dict[str, set[str]] = {}
    for key, values, _ in keys:
        for value in values:
            owners.setdefault(value, set()).add(key)
    plain_resolved = _plain(resolved)
    for sentence in re.split(r"(?<=[.!?;])\s+(?=[^\d])|\n+", resolved):
        ps = _plain(sentence)
        mentions = sorted((ps.find(alias), key) for key, _, _ in keys for alias in _aliases(key, key_names)
                          if alias and ps.find(alias) >= 0)
        if not mentions:
            continue
        cuts = [pos for pos, _ in mentions] + [len(ps)]
        for i, (pos, key) in enumerate(mentions):
            start = 0 if i == 0 else pos
            window = extract_facts(_resolve_approximations(sentence[start:cuts[i + 1]], verified)[0]).values
            for value in window:
                if value in owners and key not in owners[value]:
                    failures.append(f"cifra {value} atribuida a «{key}»")

    ranked = sorted((rank, key) for key, _, rank in keys if rank is not None)
    if ranked and SUPERLATIVE_RE.search(plain_resolved):
        found = [(plain_resolved.find(alias), key) for key, _, _ in keys for alias in _aliases(key, key_names)
                 if plain_resolved.find(alias) >= 0]
        first = min(found) if found else None
        if first is not None and first[1] != ranked[0][1]:
            failures.append(f"declara como primera a «{first[1]}» y la primera es «{ranked[0][1]}»")
    if CAUSAL_RE.search(plain_resolved) and not CAUSAL_RE.search(_plain(verified)):
        failures.append("afirma causas que los datos no registran")
    if SQL_RE.search(cand):
        failures.append("contiene SQL o código")
    if INJECTION_RE.search(cand):
        failures.append("reproduce instrucciones inyectadas")
    return ValidationResult(not failures, sorted(set(failures)))


# --- Long policy answers: the model writes a short summary, the app shows the cited text ---

def validate_policy_summary(*, question: str, verified: str, summary: str, allowed_sources: set[str],
                            conflicts: list[dict] | None = None) -> ValidationResult:
    """A 2-4 sentence summary of a long, cited policy answer. The full verified text
    (every condition and citation) is shown right below it, so the summary may omit
    details; it may not add figures, dates, citations, documents or codes, present the
    losing side of a contradiction as the rule, or pick a winner the source did not."""
    failures: list[str] = []
    cand = summary.strip()
    if not cand:
        return ValidationResult(False, ["respuesta vacía"])
    if len(cand) > 1200:
        failures.append("resumen demasiado largo")
    v, q, c = extract_facts(verified), extract_facts(question), extract_facts(cand)
    extra = set(c.values) - v.values - q.values
    if extra:
        failures.append("cifras o fechas nuevas: " + ", ".join(sorted(extra)))
    invented = sorted(c.citations - v.citations)
    if invented:
        failures.append("citas inexistentes: " + " ".join(invented))
    unknown_files = sorted(c.files - v.files - {s.lower() for s in allowed_sources})
    if unknown_files:
        failures.append("fuentes inexistentes: " + ", ".join(unknown_files))
    new_codes = sorted(c.codes - v.codes)
    if new_codes:
        failures.append("códigos de documento nuevos: " + ", ".join(new_codes))

    def prevails(doc: str) -> bool:
        return bool(re.search(r"(?<!no )(?<!no se )prevalece\s+(?:el\s+|la\s+)?" + re.escape(doc), cand, re.IGNORECASE))

    p_cand = _plain(cand)
    for conflict in conflicts or []:
        preferred = conflict.get("preferred_document")
        claims = conflict.get("claims", [])
        docs = [cl.get("document", "") for cl in claims if cl.get("document")]
        if conflict.get("resolution") == "resolved_by_source" and preferred:
            for doc in docs:
                if doc != preferred and prevails(doc):
                    failures.append(f"declara que prevalece {doc}, contra la fuente")
            # A figure that only the losing document states (FAQ: 15 días) may appear
            # only together with the governing figure and the precedence.
            losing = set().union(*[extract_facts(" ".join(cl.get("values") or [])).values
                                   for cl in claims if cl.get("document") != preferred] or [set()])
            winning = set().union(*[extract_facts(" ".join(cl.get("values") or [])).values
                                    for cl in claims if cl.get("document") == preferred] or [set()])
            if (losing - winning) & c.values:
                if not (winning <= c.values and ("prevalece" in p_cand or "contradic" in p_cand)):
                    failures.append("presenta la versión que no prevalece sin la que sí prevalece")
        elif re.search(r"(?<!no )prevalece\s+(?:el\s+|la\s+)?[\w-]+\.pdf", cand, re.IGNORECASE):
            failures.append("elige un documento ganador en una contradicción sin resolver")
    if SQL_RE.search(cand):
        failures.append("contiene SQL o código")
    if INJECTION_RE.search(cand):
        failures.append("reproduce instrucciones inyectadas")
    return ValidationResult(not failures, sorted(set(failures)))
