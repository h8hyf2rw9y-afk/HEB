"""Conservative routing for returns and receipt-related follow-ups.

No policy is synthesized here. Only policy search can supply substantive rules;
a bad retrieval cannot be displayed as a definitive company procedure.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


_RECEIPT = re.compile(r"\b(?:sin (?:el )?ticket|no (?:tiene|trae|presenta|encuentra|conserva) (?:el )?ticket|sin (?:el )?comprobante|sin (?:su )?recibo|ticket de compra)\b")
_RETURN = re.compile(r"\b(?:devoluc\w*|devolver|devuelve|reembols\w*|cambio de producto|devolviendo)\b")
_EXCHANGE_VS_REFUND = re.compile(
    r"\b(?:cambiar|cambio)\b.{0,100}\b(?:(?:reembols\w*|rembols\w*)|devoluc\w*)\b"
    r"|\b(?:(?:reembols\w*|rembols\w*)|devoluc\w*)\b.{0,100}\b(?:cambiar|cambio)\b"
)

_ORIGINAL_CARD = re.compile(
    r"\b(?:sin (?:la )?tarjeta|no (?:trae|tiene|presenta) (?:la )?tarjeta|"
    r"tarjeta original|tarjeta (?:de credito|de debito)|pago con tarjeta)\b"
)
_SERVICE = re.compile(r"\b(?:mesa de servicio|soporte|incidencia|averia|falla del sistema|levantar (?:un )?ticket|ticket de soporte)\b")
_GENERIC_PROCEDURE = re.compile(
    r"\b(?:proceso|procedimiento|pasos|tramite|hacer|funciona|que pasa|como)\b"
)
_SHORT_FOLLOWUP = re.compile(r"^(?:y |pero |entonces |en caso de que |si )")


@dataclass(frozen=True)
class ReturnRoute:
    kind: str  # no_receipt | general_process | needs_context | exchange_vs_refund
    query: str = ""
    product: str | None = None  # product the user already named ("licuadora")


_GENERIC_ITEMS = {"producto", "productos", "articulo", "articulos", "mercancia", "compra", "compras",
                  "algo", "cosa", "cosas", "pedido", "pedidos"}
_STOP = {"que", "cual", "cuales", "como", "con", "sin", "si", "en", "por", "para", "cuando", "y", "o", "pero",
         "el", "la", "los", "las", "un", "una", "de", "del", "al", "a", "es", "se", "le", "lo", "porque"}
_PRODUCT = re.compile(
    r"\b(?:devolver|devuelve|regresar|cambiar|devolucion(?:es)? de|cambio de)\s+"
    r"(?:(?:un|una|unos|unas|el|la|los|las|su|sus)\s+)?([a-z0-9][a-z0-9 ]*)"
)


def named_product(question: str) -> str | None:
    """Product the user already named («licuadora», «plancha de vapor»), so it is
    never asked for again. Generic words («un producto») do not count."""
    match = _PRODUCT.search(plain(question).strip(" ¿?¡!.,;:"))
    if not match:
        return None
    words = []
    for word in match.group(1).split():
        if word in _STOP or len(words) == 3:
            break
        words.append(word)
    if not words or words[0] in _GENERIC_ITEMS:
        return None
    return " ".join(words)


def route_returns(question: str, previous_question: str | None = None) -> ReturnRoute | None:
    q = plain(question).strip(" ¿?¡!.,;:")
    previous = plain(previous_question or "")
    if _SERVICE.search(q):
        return None
    if _EXCHANGE_VS_REFUND.search(q):
        return ReturnRoute(
            "exchange_vs_refund",
            "¿Cuál es la diferencia entre cambio de producto y reembolso "
            "según las definiciones y condiciones de procedimiento_devoluciones.pdf?",
        )
    # Missing the payment card is not a missing purchase receipt. Let RAG
    # retrieve section 6.1 (reimbursements without the original card).
    if _ORIGINAL_CARD.search(q):
        return None

    has_return = bool(_RETURN.search(q))
    previously_returns = bool(_RETURN.search(previous)) and not _SERVICE.search(previous)
    receipt = bool(_RECEIPT.search(q))
    short = len(q.split()) <= 12

    # A standalone "Si no tiene ticket" is ambiguous: it must not retrieve
    # service-desk instructions or silently infer a cash-register return.
    if receipt and not has_return and not (short and previously_returns):
        return ReturnRoute("needs_context")

    if receipt and (has_return or (short and previously_returns)):
        return ReturnRoute("no_receipt", "¿Puedo aceptar una devolución sin ticket?")

    if has_return and _GENERIC_PROCEDURE.search(q):
        # Steps come from the governing procedure (section 7). Deadlines and
        # requirements depend on the product category, so they are retrieved
        # only when the user named the product; otherwise only that is asked.
        product = named_product(question)
        return ReturnRoute("general_process", question.strip() if product else "", product)
    return None


def check_return_evidence(kind: str, *, status: str, answer: str, evidence: list[dict]) -> bool:
    """Fail closed rather than display an unrelated or category-specific passage."""
    if status != "answered" or not evidence:
        return False
    sources = {str(item.get("source", "")) for item in evidence}
    text = plain(answer)
    if kind == "exchange_vs_refund":
        # A definition of exchange alone does not establish whether, when,
        # or how a refund may be made. Require both concepts in the evidence.
        return (
            "procedimiento_devoluciones.pdf" in sources
            and "[procedimiento_devoluciones.pdf," in answer
            and bool(re.search(r"\bcambio\b|\bcambiar\b", text))
            and bool(re.search(r"\breembols\w*\b", text))
        )
    if kind == "no_receipt":
        return (
            "faq_gerentes_de_tienda.pdf" in sources
            and "[faq_gerentes_de_tienda.pdf," in answer
            and "vale de tienda" in text
            and "identificacion oficial" in text
            and "manual_apertura_y_cierre_tienda.pdf" not in answer
        )
    if kind == "general_process":
        # The FAQ may appear only as the losing side of a contradiction the
        # procedure itself resolves; it can never be the governing rule.
        faq_ok = ("faq_gerentes_de_tienda.pdf" not in answer
                  or "prevalece procedimiento_devoluciones.pdf" in text)
        return (
            "procedimiento_devoluciones.pdf" in sources
            and "[procedimiento_devoluciones.pdf," in answer
            and faq_ok
            and bool(re.search(r"\b(?:devoluc\w*|devolver|cambio)\b", text))
        )
    return False
