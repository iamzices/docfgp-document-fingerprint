"""Stable 512-permutation MinHash and indexed LSH keys."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Iterable

from ._native import minhash as _native_minhash

COMPONENTS = 512
BANDS = 32
ROWS = 16
_PRIME = (1 << 61) - 1
_PERMUTATION_SEED = b"docfgp-minhash-v2"
_UINT64_BYTES = 8
_BAND_KEY_VERSION = "v2"
_BAND_DIGEST_CHARS = 24


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
    return hashlib.blake2b(data, digest_size=_UINT64_BYTES).hexdigest()


def sketch(features: Iterable[str]) -> tuple[str, ...]:
    """Compute a deterministic, 512-component MinHash over unique features.

    Args:
        features: Textual shingles or their stable hexadecimal digests.

    Returns:
        Fixed-width hexadecimal values, or an empty tuple without features.
    """
    values = {_field_element(feature) for feature in features}
    if not values:
        return ()

    minima = _native_minhash(list(values), _permutations())
    return tuple(f"{minimum:016x}" for minimum in minima)


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
        rows = "".join(values[band * ROWS : (band + 1) * ROWS]).encode()
        digest = stable_digest(rows)[:_BAND_DIGEST_CHARS]
        keys.append(f"{namespace}:{_BAND_KEY_VERSION}:{band:02d}:{digest}")
    return tuple(keys)


def _field_element(feature: str) -> int:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    return int.from_bytes(digest[:_UINT64_BYTES], "big") % _PRIME


@functools.cache
def _permutations() -> tuple[tuple[int, int], ...]:
    # Derived from fixed bytes, so sketches never depend on Python's random module.
    return tuple(_permutation(index) for index in range(COMPONENTS))


def _permutation(index: int) -> tuple[int, int]:
    digest = hashlib.sha256(_PERMUTATION_SEED + index.to_bytes(2, "big")).digest()
    multiplier = int.from_bytes(digest[:_UINT64_BYTES], "big") % (_PRIME - 1) + 1
    offset = int.from_bytes(digest[_UINT64_BYTES : 2 * _UINT64_BYTES], "big") % _PRIME
    return multiplier, offset
