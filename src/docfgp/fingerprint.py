"""Coordinate one stateless document fingerprinting invocation."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path

from . import extract, ocr
from .minhash import BANDS, COMPONENTS, ROWS, band_keys, feature_digest, sketch, stable_digest
from .normalize import strict_text, tolerant_text, word_tokens
from .schemas import (
    DocumentFingerprint,
    FingerprintConfig,
    FingerprintError,
    PageHash,
    PageQuality,
    PageSource,
    Quality,
    QualityStatus,
    TextEvidence,
    _remaining_seconds,
)
from .visual import page_phash, visual_keys
from .winnowing import anchors

SCHEMA_VERSION = "2.0"
ALGORITHM_VERSION = "1.5"
WORD_SHINGLE_SIZE = 5
CHAR_SHINGLE_SIZE = 13
ANCHOR_WINDOW = 32
OCR_MIN_CONFIDENCE = 30.0
OCR_REPLACEMENT_CONFIDENCE = 60.0


def _raw_hash(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                _remaining_seconds(deadline)
                digest.update(chunk)
    except OSError as exc:
        raise FingerprintError(f"Cannot hash document: {exc}") from exc
    return digest.hexdigest()


def _shingles(tokens: tuple[str, ...]) -> Iterator[str]:
    for index in range(len(tokens) - WORD_SHINGLE_SIZE + 1):
        yield feature_digest("\x1f".join(tokens[index : index + WORD_SHINGLE_SIZE]).encode("utf-8"))


def _accept_ocr(
    native: str, recognized: str, confidence: float | None, *, rasterized: bool
) -> tuple[str | None, str | None]:
    """Return OCR text that meets the confidence rules, or a warning explaining its rejection."""
    if not recognized or confidence is None:
        return None, "OCR found no reliable text"
    if rasterized:
        novel = set(word_tokens(recognized)) - set(word_tokens(native))
        if confidence >= OCR_REPLACEMENT_CONFIDENCE and len(novel) >= 3:
            return recognized, None
        return None, "OCR did not provide confident additional text" if novel else None
    if confidence >= OCR_MIN_CONFIDENCE:
        return recognized, None
    return None, "OCR confidence was too low to use its text"


def _assemble_fingerprint(
    text: str,
    *,
    config: FingerprintConfig,
    deadline: float,
    source_format: str,
    byte_size: int,
    raw_sha256: str,
    page_quality: tuple[PageQuality, ...],
    page_hashes: tuple[PageHash, ...] = (),
    warnings: tuple[str, ...] = (),
) -> DocumentFingerprint:
    """Build shared text features and a versioned result from extracted content."""
    tokens = word_tokens(text)
    canonical = " ".join(tokens)
    word_hashes: list[str] = []
    counts: Counter[str] = Counter()
    for value in _shingles(tokens):
        if len(word_hashes) % 4096 == 0:
            _remaining_seconds(deadline)
        word_hashes.append(value)
        counts[value] += 1
        if len(counts) > config.max_features:
            raise FingerprintError("Word shingles exceed max_features")
    word_sketch = sketch(counts)
    _remaining_seconds(deadline)
    characters = tolerant_text(canonical)
    char_features = set()
    for index in range(len(characters) - CHAR_SHINGLE_SIZE + 1):
        if index % 4096 == 0:
            _remaining_seconds(deadline)
        char_features.add(feature_digest(characters[index : index + CHAR_SHINGLE_SIZE].encode()))
        if len(char_features) > config.max_features:
            raise FingerprintError("Character shingles exceed max_features")
    char_sketch = sketch(char_features)
    _remaining_seconds(deadline)
    passage_anchors = anchors(word_hashes, ANCHOR_WINDOW)
    visual_lookup = tuple(sorted({key for page in page_hashes for key in page.lookup_keys}))
    if word_sketch:
        text_evidence: TextEvidence = "word_shingles"
    elif char_sketch:
        text_evidence = "character_shingles"
    elif canonical:
        text_evidence = "exact_only"
    else:
        text_evidence = "none"
    if tokens and page_hashes:
        status: QualityStatus = "text_and_visual"
    elif tokens:
        status = "text_only"
    elif page_hashes:
        status = "visual_only"
    else:
        status = "insufficient_evidence"
    quality_warnings = list(warnings)
    if status == "visual_only":
        quality_warnings.append("No reliable text was extracted; textual matching is unavailable")
    if status == "insufficient_evidence":
        quality_warnings.append("Neither text nor nonblank visual evidence was extracted")
    if text_evidence == "exact_only":
        quality_warnings.append(
            "Text supports exact matching only; no shingle sketch was available"
        )
    elif text_evidence == "character_shingles":
        quality_warnings.append(
            "Word-shingle matching is unavailable; character retrieval remains available"
        )
    return DocumentFingerprint(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        source_format=source_format,
        byte_size=byte_size,
        page_count=len(page_quality),
        exact_hashes={
            "raw_sha256": raw_sha256,
            "canonical_sha256": stable_digest(canonical.encode()) if canonical else None,
        },
        lookup_keys={
            "word_lsh": band_keys(word_sketch, "word"),
            "char_lsh": band_keys(char_sketch, "char"),
            "passage": tuple(f"passage:v1:{value}" for value in passage_anchors),
            "visual": visual_lookup,
        },
        minhash=word_sketch,
        char_minhash=char_sketch,
        shingle_counts=tuple(sorted(counts.items())) if config.include_shingle_counts else (),
        shingle_total=sum(counts.values()),
        shingle_unique=len(counts),
        page_hashes=page_hashes,
        quality=Quality(status, text_evidence, page_quality, tuple(quality_warnings)),
        parameters={
            "word_shingle_size": WORD_SHINGLE_SIZE,
            "char_shingle_size": CHAR_SHINGLE_SIZE,
            "minhash_components": COMPONENTS,
            "lsh_bands": BANDS,
            "lsh_rows": ROWS,
            "anchor_window": ANCHOR_WINDOW,
            "visual_hash": "phash64-dct32",
            "render_dpi": config.render_dpi,
            "ocr_dpi": config.ocr_dpi,
            **ocr.provenance(config),
            "ocr_min_confidence": OCR_MIN_CONFIDENCE,
            "ocr_replacement_confidence": OCR_REPLACEMENT_CONFIDENCE,
            "raster_ocr_coverage": extract.RASTER_OCR_COVERAGE,
            "shingle_counts_included": config.include_shingle_counts,
        },
    )


class Fingerprinter:
    """Generate stateless document fingerprints.

    Args:
        config: Optional bounded processing settings.
    """

    def __init__(self, config: FingerprintConfig | None = None) -> None:
        """Store immutable processing limits after confirming the OCR models exist.

        Args:
            config: Optional processing settings.

        Raises:
            FingerprintError: A configured OCR model is not installed.
        """
        self.config = config or FingerprintConfig()
        ocr.model_directory(self.config)

    def fingerprint(self, path: str | Path) -> DocumentFingerprint:
        """Create a document's textual and visual fingerprint.

        Args:
            path: PDF, DOCX, PPTX, or supported image path.

        Returns:
            Versioned fingerprint.

        Raises:
            FingerprintError: Input is invalid, limits are exceeded, or processing fails.
        """
        deadline = time.monotonic() + self.config.max_document_seconds
        source = Path(path).expanduser()
        source_format, byte_size = extract.identify(source, self.config)
        _remaining_seconds(deadline)
        raw_sha256 = _raw_hash(source, deadline)
        page_text: list[str] = []
        page_hashes: list[PageHash] = []
        page_quality: list[PageQuality] = []
        warnings: list[str] = []
        total_text_chars = 0

        def consume(
            pages: Iterable[extract.ExtractedPage],
        ) -> None:
            nonlocal total_text_chars
            for page in pages:
                page_number = page.number
                visual = page.visual
                ocr_image = page.ocr_image
                try:
                    _remaining_seconds(deadline)
                    native = strict_text(page.native_text)
                    extracted = "" if page.ocr_reason == "invalid_native_text" else native
                    source: PageSource = "native" if extracted else "none"
                    attempted = ocr_image is not None
                    warning = None
                    confidence = None
                    if ocr_image is not None:
                        try:
                            result = ocr.read_text(ocr_image, self.config, deadline)
                            confidence = result.confidence
                            accepted, warning = _accept_ocr(
                                native,
                                strict_text(result.text),
                                confidence,
                                rasterized=page.ocr_reason == "rasterized_content",
                            )
                            if accepted:
                                extracted = accepted
                                source = "ocr"
                        except FingerprintError as exc:
                            warning = str(exc)
                    if warning:
                        warnings.append(f"page {page_number}: {warning}")
                    page_text.append(extracted)
                    total_text_chars += len(extracted)
                    if total_text_chars > self.config.max_text_chars:
                        raise FingerprintError("Extracted text exceeds max_text_chars")
                    value = page_phash(visual)
                    if value is not None:
                        page_hashes.append(PageHash(page_number, value, visual_keys(value)))
                    page_quality.append(
                        PageQuality(
                            page=page_number,
                            source=source,
                            text_chars=len(extracted),
                            native_text_chars=len(native),
                            ocr_attempted=attempted,
                            ocr_reason=page.ocr_reason,
                            ocr_confidence=confidence,
                            warning=warning,
                        )
                    )
                    _remaining_seconds(deadline)
                except (OSError, ValueError) as exc:
                    raise FingerprintError(f"Cannot process page {page_number}: {exc}") from exc
                finally:
                    visual.close()
                    if ocr_image is not None and ocr_image is not visual:
                        ocr_image.close()

        if source_format not in {"pdf", "docx", "pptx"}:
            with closing(extract.image_pages(source, self.config, deadline)) as image_frames:
                consume(image_frames)
        else:
            with extract.pdf_path(source, source_format, self.config, deadline) as pdf:
                with closing(extract.pdf_pages(pdf, self.config, deadline)) as pdf_frames:
                    consume(pdf_frames)

        return _assemble_fingerprint(
            " ".join(page_text),
            config=self.config,
            deadline=deadline,
            source_format=source_format,
            byte_size=byte_size,
            raw_sha256=raw_sha256,
            page_quality=tuple(page_quality),
            page_hashes=tuple(page_hashes),
            warnings=tuple(warnings),
        )
