"""Coordinate one stateless document fingerprinting invocation."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import extract, ocr
from .minhash import BANDS, COMPONENTS, ROWS, band_keys, feature_digest, sketch, stable_digest
from .normalize import strict_text, tolerant_text, word_tokens
from .schemas import (
    Box,
    Deadline,
    DocumentFingerprint,
    FingerprintConfig,
    FingerprintError,
    PageHash,
    PageQuality,
    PageSource,
    Quality,
    QualityStatus,
    TextEvidence,
    converted_errors,
)
from .visual import page_phash, visual_keys
from .winnowing import anchors

SCHEMA_VERSION = "2.1"
ALGORITHM_VERSION = "1.6"
WORD_SHINGLE_SIZE = 5
CHAR_SHINGLE_SIZE = 13
ANCHOR_WINDOW = 32
OCR_MIN_CONFIDENCE = 30.0
OCR_ADDITION_CONFIDENCE = 60.0
_OCR_MIN_ADDED_WORDS = 3
_HASH_CHUNK_BYTES = 1024 * 1024
_DEADLINE_CHECK_INTERVAL = 4096
_SHINGLE_SEPARATOR = "\x1f"
_NO_OCR_TEXT = "OCR found no reliable text"
_STATUS_WARNINGS: dict[QualityStatus, str] = {
    "visual_only": "No reliable text was extracted; textual matching is unavailable",
    "insufficient_evidence": "Neither text nor nonblank visual evidence was extracted",
}
_TEXT_EVIDENCE_WARNINGS: dict[TextEvidence, str] = {
    "exact_only": "Text supports exact matching only; no shingle sketch was available",
    "character_shingles": (
        "Word-shingle matching is unavailable; character retrieval remains available"
    ),
}


@dataclass(frozen=True, slots=True)
class _OCROutcome:
    """OCR text accepted for a page, with its confidence or the reason it was not used."""

    text: str | None = None
    confidence: float | None = None
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class _PageEvidence:
    text: str
    quality: PageQuality
    page_hash: PageHash | None


@dataclass(frozen=True, slots=True)
class _DocumentEvidence:
    source_format: str
    byte_size: int
    raw_sha256: str
    pages: tuple[_PageEvidence, ...]


@dataclass(frozen=True, slots=True)
class _WordFeatures:
    hashes: tuple[str, ...]
    counts: Counter[str]
    sketch: tuple[str, ...]


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
        deadline = Deadline(self.config.max_document_seconds)
        document_path = Path(path).expanduser()
        source_format, byte_size = extract.identify(document_path, self.config)
        deadline.check()

        document = _DocumentEvidence(
            source_format=source_format,
            byte_size=byte_size,
            raw_sha256=_file_sha256(document_path, deadline),
            pages=_read_pages(document_path, source_format, self.config, deadline),
        )

        return _assemble_fingerprint(document, self.config, deadline)


def _file_sha256(path: Path, deadline: Deadline) -> str:
    digest = hashlib.sha256()
    with converted_errors("Cannot hash document"), path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            deadline.check()
            digest.update(chunk)
    return digest.hexdigest()


def _read_pages(
    path: Path, source_format: str, config: FingerprintConfig, deadline: Deadline
) -> tuple[_PageEvidence, ...]:
    evidence = []
    text_chars = 0

    with (
        closing(ocr.Reader(config)) as reader,
        extract.pages(path, source_format, config, deadline) as pages,
    ):
        for page in pages:
            deadline.check()
            with closing(page), converted_errors(f"Cannot process page {page.number}"):
                page_evidence = _read_page(page, reader, deadline)

            text_chars += len(page_evidence.text)
            if text_chars > config.max_text_chars:
                raise FingerprintError("Extracted text exceeds max_text_chars")
            evidence.append(page_evidence)
            deadline.check()

    return tuple(evidence)


def _read_page(
    page: extract.ExtractedPage, reader: ocr.Reader, deadline: Deadline
) -> _PageEvidence:
    native_text = strict_text(page.native_text)
    usable_text = "" if page.ocr_reason == "invalid_native_text" else native_text

    outcome = _OCROutcome()
    if page.ocr_image is not None:
        outcome = _ocr_outcome(page, page.ocr_image, reader, deadline)

    text, page_source = _combine_text(usable_text, outcome.text)
    quality = PageQuality(
        page=page.number,
        source=page_source,
        text_chars=len(text),
        native_text_chars=len(native_text),
        ocr_attempted=page.ocr_image is not None,
        ocr_reason=page.ocr_reason,
        ocr_confidence=outcome.confidence,
        warning=outcome.warning,
    )

    phash = page_phash(page.visual)
    page_hash = PageHash(page.number, phash, visual_keys(phash)) if phash is not None else None
    return _PageEvidence(text, quality, page_hash)


def _ocr_outcome(
    page: extract.ExtractedPage, image: Image.Image, reader: ocr.Reader, deadline: Deadline
) -> _OCROutcome:
    try:
        result = reader.read_text(image, deadline)
    except FingerprintError as exc:
        return _OCROutcome(warning=str(exc))

    if page.ocr_reason == "rasterized_content":
        return _ocr_addition(result, page.native_text_boxes)
    return _ocr_replacement(result)


def _ocr_replacement(result: ocr.OCRResult) -> _OCROutcome:
    text = strict_text(result.text)
    if not text or result.confidence is None:
        return _OCROutcome(warning=_NO_OCR_TEXT)
    if result.confidence < OCR_MIN_CONFIDENCE:
        return _OCROutcome(
            confidence=result.confidence, warning="OCR confidence was too low to use its text"
        )
    return _OCROutcome(text, result.confidence)


def _ocr_addition(result: ocr.OCRResult, native_text_boxes: Sequence[Box]) -> _OCROutcome:
    if not strict_text(result.text) or result.confidence is None:
        return _OCROutcome(warning=_NO_OCR_TEXT)

    added = [word for word in result.words if not _is_covered(word.box, native_text_boxes)]
    if not added:
        return _OCROutcome()

    confidence = ocr.mean_confidence(added)
    text = strict_text(_join_lines(added))
    if confidence < OCR_ADDITION_CONFIDENCE or len(word_tokens(text)) < _OCR_MIN_ADDED_WORDS:
        return _OCROutcome(
            confidence=confidence, warning="OCR did not provide confident additional text"
        )
    return _OCROutcome(text, confidence)


def _is_covered(box: Box, native_text_boxes: Sequence[Box]) -> bool:
    return any(native_box.contains_center_of(box) for native_box in native_text_boxes)


def _join_lines(words: Sequence[ocr.OCRWord]) -> str:
    lines: dict[ocr.LinePosition, list[str]] = {}
    for word in words:
        lines.setdefault(word.line, []).append(word.text)
    return "\n".join(" ".join(line) for line in lines.values())


def _combine_text(native_text: str, ocr_text: str | None) -> tuple[str, PageSource]:
    if ocr_text is None:
        return native_text, "native" if native_text else "none"
    if native_text:
        return f"{native_text} {ocr_text}", "native_and_ocr"
    return ocr_text, "ocr"


def _assemble_fingerprint(
    document: _DocumentEvidence, config: FingerprintConfig, deadline: Deadline
) -> DocumentFingerprint:
    tokens = word_tokens(" ".join(page.text for page in document.pages))
    canonical = " ".join(tokens)
    words = _word_features(tokens, deadline)
    char_sketch = _char_sketch(canonical, deadline)
    page_hashes = tuple(page.page_hash for page in document.pages if page.page_hash is not None)
    passages = anchors(words.hashes, ANCHOR_WINDOW)

    return DocumentFingerprint(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        source_format=document.source_format,
        byte_size=document.byte_size,
        page_count=len(document.pages),
        exact_hashes={
            "raw_sha256": document.raw_sha256,
            "canonical_sha256": stable_digest(canonical.encode()) if canonical else None,
        },
        lookup_keys={
            "word_lsh": band_keys(words.sketch, "word"),
            "char_lsh": band_keys(char_sketch, "char"),
            "passage": tuple(f"passage:v1:{passage}" for passage in passages),
            "visual": _visual_lookup_keys(page_hashes),
        },
        minhash=words.sketch,
        char_minhash=char_sketch,
        shingle_counts=tuple(sorted(words.counts.items())) if config.include_shingle_counts else (),
        shingle_total=sum(words.counts.values()),
        shingle_unique=len(words.counts),
        page_hashes=page_hashes,
        quality=_quality(document.pages, words.sketch, char_sketch, canonical),
        parameters=_parameters(config),
    )


def _word_features(tokens: tuple[str, ...], deadline: Deadline) -> _WordFeatures:
    hashes = []
    for index, shingle_hash in enumerate(_shingles(tokens)):
        if index % _DEADLINE_CHECK_INTERVAL == 0:
            deadline.check()
        hashes.append(shingle_hash)

    counts = Counter(hashes)
    word_sketch = sketch(counts)
    deadline.check()
    return _WordFeatures(tuple(hashes), counts, word_sketch)


def _shingles(tokens: tuple[str, ...]) -> Iterator[str]:
    for index in range(len(tokens) - WORD_SHINGLE_SIZE + 1):
        shingle = _SHINGLE_SEPARATOR.join(tokens[index : index + WORD_SHINGLE_SIZE])
        yield feature_digest(shingle.encode("utf-8"))


def _char_sketch(canonical: str, deadline: Deadline) -> tuple[str, ...]:
    characters = tolerant_text(canonical)
    features = set()
    for index in range(len(characters) - CHAR_SHINGLE_SIZE + 1):
        if index % _DEADLINE_CHECK_INTERVAL == 0:
            deadline.check()
        features.add(feature_digest(characters[index : index + CHAR_SHINGLE_SIZE].encode()))

    char_sketch = sketch(features)
    deadline.check()
    return char_sketch


def _visual_lookup_keys(page_hashes: Sequence[PageHash]) -> tuple[str, ...]:
    keys: set[str] = set()
    for page_hash in page_hashes:
        keys.update(page_hash.lookup_keys)
    return tuple(sorted(keys))


def _quality(
    pages: Sequence[_PageEvidence],
    word_sketch: tuple[str, ...],
    char_sketch: tuple[str, ...],
    canonical: str,
) -> Quality:
    text_evidence = _text_evidence(word_sketch, char_sketch, canonical)
    has_visual = any(page.page_hash is not None for page in pages)
    status = _status(has_text=bool(canonical), has_visual=has_visual)

    warnings = [
        f"page {page.quality.page}: {page.quality.warning}"
        for page in pages
        if page.quality.warning
    ]
    status_warnings = (_STATUS_WARNINGS.get(status), _TEXT_EVIDENCE_WARNINGS.get(text_evidence))
    warnings.extend(warning for warning in status_warnings if warning is not None)

    return Quality(status, text_evidence, tuple(page.quality for page in pages), tuple(warnings))


def _text_evidence(
    word_sketch: tuple[str, ...], char_sketch: tuple[str, ...], canonical: str
) -> TextEvidence:
    if word_sketch:
        return "word_shingles"
    if char_sketch:
        return "character_shingles"
    if canonical:
        return "exact_only"
    return "none"


def _status(*, has_text: bool, has_visual: bool) -> QualityStatus:
    if has_text and has_visual:
        return "text_and_visual"
    if has_text:
        return "text_only"
    if has_visual:
        return "visual_only"
    return "insufficient_evidence"


def _parameters(config: FingerprintConfig) -> dict[str, int | float | str | bool]:
    return {
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
        "ocr_addition_confidence": OCR_ADDITION_CONFIDENCE,
        "raster_ocr_coverage": extract.RASTER_OCR_COVERAGE,
        "shingle_counts_included": config.include_shingle_counts,
    }
