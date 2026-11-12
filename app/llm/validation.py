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
