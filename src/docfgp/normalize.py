"""Canonicalize extracted text while retaining significant numbers."""

from __future__ import annotations

import re
import unicodedata

_HYPHENATION = re.compile(r"(?<=\w)[-\u00ad][ \t]*\r?\n[ \t]*(?=\w)")
_WHITESPACE = re.compile(r"\s+")
_WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def strict_text(text: str) -> str:
    """Normalize Unicode, soft line breaks, and whitespace.

    Args:
        text: Raw extracted page text.

    Returns:
        Canonical casefolded text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHENATION.sub("", text)
    text = text.replace("\u00ad", "")
    return _WHITESPACE.sub(" ", text).strip().casefold()


def word_tokens(text: str) -> tuple[str, ...]:
    """Extract Unicode word tokens from normalized text.

    Args:
        text: Canonical text.

    Returns:
        Tokens in reading order.
    """
    return tuple(_WORDS.findall(text))


def tolerant_text(text: str) -> str:
    """Remove punctuation and spacing for OCR-tolerant character retrieval.

    Args:
        text: Canonical text.

    Returns:
        Alphanumeric Unicode characters, including digits.
    """
    return "".join(ch for ch in text if ch.isalnum())
