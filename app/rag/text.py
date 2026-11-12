"""Spanish text normalization and tokenization for lexical retrieval."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9¿¡•\-])")

STOPWORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes aqui asi aun cada como con contra
    cual cuales cuando cuanto cuantos cuantas de del desde donde dos e el ella ellas ellos en
    entre era eran es esa esas ese eso esos esta estan estas este esto estos fue fueron ha han
    hasta hay la las le les lo los mas me mi mis muy nos o os otra otras otro otros para pero
    poco por porque puede pueden que quien quienes se segun ser si sin sino sobre son su sus
    tal tambien tan tanto te tiene tienen todo todos tu tus u un una unas uno unos usted ustedes
    y ya yo cual es hacer hace debo debe deben esta estoy segun sea son sus
    politica politicas
    puedo podemos tengo tenemos tarda tardan tardar necesito necesita quiero saber dime explica
    hago hacemos existe existen aplica cuanta cuantas quien quienes cual cuales donde cuando como""".split()
)
# "política" and question fillers ("puedo", "tengo", "tarda"...) carry no topical
# signal; keeping them would make abstention penalize ordinary phrasing.


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    return strip_accents(text).lower()


def stem(token: str) -> str:
    """Very light Spanish stemmer: plural stripping + prefix truncation.

    Deterministic and dependency-free; good enough for BM25 over short policies.
    """
    if token.isdigit() or any(ch.isdigit() for ch in token):
        return token
    if len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token[:6]


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize(text))
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return [stem(t) for t in tokens]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
