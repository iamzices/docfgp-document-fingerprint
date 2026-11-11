"""Stable 512-permutation MinHash and indexed LSH keys."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

COMPONENTS = 512
BANDS = 32
ROWS = 16
_PRIME = (1 << 61) - 1


def _coefficients(index: int) -> tuple[int, int]:
    # Derive every permutation from bytes, independent of Python's random module.
    digest = hashlib.sha256(b"docfgp-minhash-v2" + index.to_bytes(2, "big")).digest()
    return int.from_bytes(digest[:8], "big") % (_PRIME - 1) + 1, int.from_bytes(
        digest[8:16], "big"
    ) % _PRIME


_COEFFICIENTS = tuple(_coefficients(index) for index in range(COMPONENTS))


def stable_digest(data: bytes) -> str:
    """Return a stable SHA-256 digest.

    Args:
        data: Input bytes.

    Returns:
        Lowercase hexadecimal digest.
    """
    return hashlib.sha256(data).hexdigest()


def feature_digest(data: bytes) -> str:
    """Return a stable 64-bit BLAKE2b digest for a text feature.

    Args:
        data: Input bytes.

    Returns:
        Lowercase hexadecimal digest.
    """
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def sketch(features: Iterable[str]) -> tuple[str, ...]:
    """Compute a deterministic, 512-component MinHash over unique features.

    Args:
        features: Textual shingles or their stable hexadecimal digests.

    Returns:
        Fixed-width hexadecimal values, or an empty tuple without features.
    """
    minima = [_PRIME] * COMPONENTS
    seen: set[str] = set()
    for feature in features:
        if feature in seen:
            continue
        seen.add(feature)
        x = int.from_bytes(hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big") % _PRIME
        for index, (a, b) in enumerate(_COEFFICIENTS):
            value = (a * x + b) % _PRIME
            if value < minima[index]:
                minima[index] = value
    return tuple(f"{value:016x}" for value in minima) if seen else ()


def band_keys(values: tuple[str, ...], namespace: str) -> tuple[str, ...]:
    """Build 32 stable keys from consecutive groups of 16 components.

    Args:
        values: A complete MinHash sketch.
        namespace: Word or character sketch identifier.

    Returns:
        Indexed keys, or no keys for an empty sketch.

    Raises:
        ValueError: The sketch has the wrong length.
    """
    if not values:
        return ()
    if len(values) != COMPONENTS:
        raise ValueError("A banded sketch must contain 512 components")
    keys = []
    for band in range(BANDS):
        data = "".join(values[band * ROWS : (band + 1) * ROWS]).encode()
        keys.append(f"{namespace}:v2:{band:02d}:{stable_digest(data)[:24]}")
    return tuple(keys)
