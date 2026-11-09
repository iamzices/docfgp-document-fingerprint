"""Generate CPU perceptual hashes and indexed keys for page images."""

from __future__ import annotations

import math
import statistics
from typing import cast

from PIL import Image, ImageOps

_COSINES = tuple(
    tuple(math.cos(math.pi * (2 * x + 1) * frequency / 64) for x in range(32))
    for frequency in range(8)
)


def normalized_grayscale(image: Image.Image) -> Image.Image:
    """Apply orientation and transparency before visual analysis.

    Args:
        image: One decoded page or image frame.

    Returns:
        Grayscale image with transparent pixels composited on white.
    """
    with ImageOps.exif_transpose(image) as oriented:
        if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
            with oriented.convert("RGBA") as rgba, Image.new("RGBA", rgba.size, "white") as white:
                with Image.alpha_composite(white, rgba) as composited:
                    return composited.convert("L")
        return oriented.convert("L")


def page_phash(image: Image.Image) -> str | None:
    """Hash a nonblank page after consistent visual normalization.

    Args:
        image: One decoded page or image frame.

    Returns:
        A 64-bit hash as 16 hexadecimal digits, or None for a blank page.
    """
    with normalized_grayscale(image) as grayscale:
        low, high = cast(tuple[int, int], grayscale.getextrema())
        if high - low < 5:
            return None
        with grayscale.resize((32, 32), Image.Resampling.LANCZOS) as reduced:
            pixels = list(reduced.tobytes())
    rows = [pixels[offset : offset + 32] for offset in range(0, 1024, 32)]
    horizontal = [
        [sum(row[x] * _COSINES[u][x] for x in range(32)) for u in range(8)] for row in rows
    ]
    coefficients = [
        sum(horizontal[y][u] * _COSINES[v][y] for y in range(32))
        for v in range(8)
        for u in range(8)
    ]
    median = statistics.median(coefficients[1:])
    bits = sum(1 << index for index, value in enumerate(coefficients) if value > median)
    return f"{bits:016x}"


def visual_keys(value: str) -> tuple[str, ...]:
    """Split a 64-bit page hash into four indexed chunks.

    Args:
        value: Page hash returned by `page_phash`.

    Returns:
        Four versioned lookup keys.
    """
    return tuple(f"visual:v1:{index}:{value[index * 4 : (index + 1) * 4]}" for index in range(4))
