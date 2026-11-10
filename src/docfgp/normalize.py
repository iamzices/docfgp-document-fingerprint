"""Canonicalize extracted text while retaining significant numbers."""

from __future__ import annotations

import unicodedata

import regex

_HYPHENATION = regex.compile(r"(?<=\w)[-\u00ad][ \t]*\r?\n[ \t]*(?=\w)")
_WHITESPACE = regex.compile(r"\s+")
# Scripts written without spaces between words; each letter or digit is one token.
_UNSPACED = r"\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}\p{Tibetan}"
_WORD = rf"[\w--_--{_UNSPACED}]"
_WORDS = regex.compile(
    rf"[[\p{{L}}\p{{N}}]&&[{_UNSPACED}]]\p{{M}}*|{_WORD}+(?:['’]{_WORD}+)?", regex.VERSION1
)


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

    Words keep their combining marks. Scripts without word spacing yield one
    token per letter or digit.

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
        Alphanumeric Unicode characters and combining marks, including digits.
    """
    return "".join(ch for ch in text if ch.isalnum() or unicodedata.category(ch).startswith("M"))
