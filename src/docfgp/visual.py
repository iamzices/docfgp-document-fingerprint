"""Generate CPU perceptual hashes and indexed keys for page images."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import cast

from PIL import Image, ImageOps

_DCT_SIZE = 32
_KEPT_FREQUENCIES = 8
_MIN_BLANK_CONTRAST = 5
_KEY_CHUNKS = 4
_COSINES = tuple(
    tuple(math.cos(math.pi * (2 * x + 1) * frequency / (2 * _DCT_SIZE)) for x in range(_DCT_SIZE))
    for frequency in range(_KEPT_FREQUENCIES)
)


def normalized_grayscale(image: Image.Image) -> Image.Image:
    """Apply orientation and transparency before visual analysis.

    Args:
        image: One decoded page or image frame, oriented in place.

    Returns:
        Grayscale image with transparent pixels composited on white.
    """
    ImageOps.exif_transpose(image, in_place=True)

    if image.mode not in {"RGBA", "LA"} and "transparency" not in image.info:
        return image.convert("L")
    with (
        image.convert("RGBA") as rgba,
        Image.new("RGBA", rgba.size, "white") as background,
        Image.alpha_composite(background, rgba) as composited,
    ):
        return composited.convert("L")


def page_phash(image: Image.Image) -> str | None:
    """Hash a nonblank page.

    Args:
        image: One page or image frame from `normalized_grayscale`.

    Returns:
        A 64-bit hash as 16 hexadecimal digits, or None for a blank page.
    """
    darkest, lightest = cast(tuple[int, int], image.getextrema())
    if lightest - darkest < _MIN_BLANK_CONTRAST:
        return None

    with image.resize((_DCT_SIZE, _DCT_SIZE), Image.Resampling.LANCZOS) as reduced:
        pixels = list(reduced.tobytes())

    coefficients = _low_frequency_dct(pixels)

    # The DC coefficient is left out of the median, as in the standard pHash.
    median = statistics.median(coefficients[1:])
    bits = 0
    for index, coefficient in enumerate(coefficients):
        if coefficient > median:
            bits |= 1 << index
    return f"{bits:016x}"


def visual_keys(phash: str) -> tuple[str, ...]:
    """Split a 64-bit page hash into four indexed chunks.

    Args:
        phash: Page hash returned by `page_phash`.

    Returns:
        Four versioned lookup keys.
    """
    width = len(phash) // _KEY_CHUNKS
    return tuple(
        f"visual:v1:{chunk}:{phash[chunk * width : (chunk + 1) * width]}"
        for chunk in range(_KEY_CHUNKS)
    )


def _low_frequency_dct(pixels: list[int]) -> list[float]:
    # A separable 2-D DCT: transform every row, then the columns of the kept frequencies.
    row_frequencies = []
    for row_start in range(0, _DCT_SIZE * _DCT_SIZE, _DCT_SIZE):
        row = pixels[row_start : row_start + _DCT_SIZE]
        row_frequencies.append([_dot(row, _COSINES[u]) for u in range(_KEPT_FREQUENCIES)])

    coefficients = []
    for v in range(_KEPT_FREQUENCIES):
        for u in range(_KEPT_FREQUENCIES):
            column = [row_frequencies[y][u] for y in range(_DCT_SIZE)]
            coefficients.append(_dot(column, _COSINES[v]))
    return coefficients


def _dot(values: Sequence[float], weights: Sequence[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights, strict=True))
