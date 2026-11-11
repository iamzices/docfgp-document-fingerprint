"""Versioned public schemas and processing limits for document fingerprints."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

QualityStatus = Literal["text_and_visual", "text_only", "visual_only", "insufficient_evidence"]
TextEvidence = Literal["word_shingles", "character_shingles", "exact_only", "none"]
PageSource = Literal["native", "ocr", "native_and_ocr", "none"]
OCRReason = Literal[
    "missing_native_text", "invalid_native_text", "rasterized_content", "image_input"
]


class FingerprintError(Exception):
    """Report invalid input, unavailable dependencies, or incomplete processing."""


def remaining_seconds(deadline: float) -> float:
    """Return the time left before a document deadline.

    Args:
        deadline: Deadline on the `time.monotonic` clock.

    Returns:
        Positive seconds remaining.

    Raises:
        FingerprintError: The deadline has passed.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FingerprintError("Document exceeded max_document_seconds")
    return remaining


@dataclass(frozen=True)
class FingerprintConfig:
    """Configure bounded document processing.

    Attributes:
        max_file_bytes: Maximum original input size.
        max_pages: Maximum PDF pages or image frames.
        max_page_pixels: Maximum pixels rendered for one PDF page.
        max_image_pixels: Maximum pixels decoded for one image frame.
        max_text_chars: Maximum total extracted characters.
        render_dpi: Visual rendering resolution, bounded by max_page_pixels.
        ocr_dpi: OCR rendering resolution, bounded by max_page_pixels.
        process_timeout: Seconds allowed for each conversion or OCR call.
        max_document_seconds: Cooperative deadline checked between processing stages.
        tessdata_path: Optional directory of Tesseract language models.
        ocr_language: Installed Tesseract language codes.
        include_shingle_counts: Include sorted exact occurrence counts.
    """

    max_file_bytes: int = 50_000_000
    max_pages: int = 200
    max_page_pixels: int = 8_000_000
    max_image_pixels: int = 40_000_000
    max_text_chars: int = 2_000_000
    render_dpi: int = 96
    ocr_dpi: int = 180
    process_timeout: float = 60.0
    max_document_seconds: float = 180.0
    tessdata_path: str | Path | None = None
    ocr_language: str = "eng"
    include_shingle_counts: bool = True

    def __post_init__(self) -> None:
        """Reject limits that would make bounded processing impossible.

        Raises:
            ValueError: A limit or resolution is invalid.
        """
        values = (
            self.max_file_bytes,
            self.max_pages,
            self.max_page_pixels,
            self.max_image_pixels,
            self.max_text_chars,
            self.render_dpi,
            self.ocr_dpi,
        )
        durations = (self.process_timeout, self.max_document_seconds)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values
        ):
            raise ValueError("All integer processing limits must be positive")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in durations
        ):
            raise ValueError("All processing limits must be positive")
        if self.render_dpi > 300 or self.ocr_dpi > 300:
            raise ValueError("Rendering resolution cannot exceed 300 DPI")
        if not isinstance(self.ocr_language, str) or not self.ocr_language.strip():
            raise ValueError("ocr_language must not be empty")
        if self.tessdata_path is not None:
            if not isinstance(self.tessdata_path, (str, Path)):
                raise ValueError("tessdata_path must be a path")
            object.__setattr__(self, "tessdata_path", str(self.tessdata_path))


@dataclass(frozen=True)
class PageQuality:
    """Record extraction provenance for one page or image frame."""

    page: int
    source: PageSource
    text_chars: int
    native_text_chars: int
    ocr_attempted: bool
    ocr_reason: OCRReason | None = None
    ocr_confidence: float | None = None
    warning: str | None = None


@dataclass(frozen=True)
class Quality:
    """State whether the fingerprint carries text, visual, or insufficient evidence."""

    status: QualityStatus
    text_evidence: TextEvidence
    pages: tuple[PageQuality, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PageHash:
    """Store a page perceptual hash and its indexed visual keys."""

    page: int
    phash: str
    lookup_keys: tuple[str, ...]


@dataclass(frozen=True)
class DocumentFingerprint:
    """Serializable, versioned document fingerprint."""

    schema_version: str
    algorithm_version: str
    source_format: str
    byte_size: int
    page_count: int
    exact_hashes: Mapping[str, str | None]
    lookup_keys: Mapping[str, tuple[str, ...]]
    minhash: tuple[str, ...]
    char_minhash: tuple[str, ...]
    shingle_counts: tuple[tuple[str, int], ...]
    shingle_total: int
    shingle_unique: int
    page_hashes: tuple[PageHash, ...]
    quality: Quality
    parameters: Mapping[str, int | float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze mapping fields so the fingerprint cannot change after creation."""
        object.__setattr__(self, "exact_hashes", MappingProxyType(dict(self.exact_hashes)))
        object.__setattr__(
            self,
            "lookup_keys",
            MappingProxyType({name: tuple(keys) for name, keys in self.lookup_keys.items()}),
        )
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation.

        Returns:
            Recursively serialized fingerprint with stable field order.
        """
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "source_format": self.source_format,
            "byte_size": self.byte_size,
            "page_count": self.page_count,
            "exact_hashes": dict(self.exact_hashes),
            "lookup_keys": {name: list(keys) for name, keys in self.lookup_keys.items()},
            "minhash": list(self.minhash),
            "char_minhash": list(self.char_minhash),
            "shingle_counts": [[value, count] for value, count in self.shingle_counts],
            "shingle_total": self.shingle_total,
            "shingle_unique": self.shingle_unique,
            "page_hashes": [
                {"page": item.page, "phash": item.phash, "lookup_keys": list(item.lookup_keys)}
                for item in self.page_hashes
            ],
            "quality": {
                "status": self.quality.status,
                "text_evidence": self.quality.text_evidence,
                "pages": [
                    {
                        "page": page.page,
                        "source": page.source,
                        "text_chars": page.text_chars,
                        "native_text_chars": page.native_text_chars,
                        "ocr_attempted": page.ocr_attempted,
                        "ocr_reason": page.ocr_reason,
                        "ocr_confidence": page.ocr_confidence,
                        "warning": page.warning,
                    }
                    for page in self.quality.pages
                ],
                "warnings": list(self.quality.warnings),
            },
            "parameters": dict(self.parameters),
        }

    def to_json(self) -> str:
        """Serialize the fingerprint as compact JSON.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
