"""Run bounded Tesseract OCR and retain word confidence evidence."""

from __future__ import annotations

import csv
import functools
import io
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import tesserocr
from PIL import Image

from .schemas import FingerprintConfig, FingerprintError, remaining_seconds

_TESSDATA = Path(__file__).with_name("tessdata")
_MANIFEST = json.loads((_TESSDATA / "models.json").read_text(encoding="utf-8"))
_LATIN_LANGUAGES = frozenset(
    "afr aze bos bre cat ceb ces cos cym dan deu epo est eus fao fil fin fra fry gla gle glg hat "
    "hrv hun ind isl ita jav lat lav lit ltz mlt mri msa nld nor oci pol por que ron slk slv spa "
    "sqi srp_latn sun swa swe ton tur uzb yor".split()
)
_TSV_FIELDS = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)


@dataclass(frozen=True)
class OCRWord:
    """Hold one recognized word with its pixel bounds and block, paragraph, and line."""

    text: str
    confidence: float
    bounds: tuple[int, int, int, int]
    line: tuple[int, int, int]


@dataclass(frozen=True)
class OCRResult:
    """Hold recognized text, its words, and mean confidence over recognized words."""

    text: str
    confidence: float | None
    words: tuple[OCRWord, ...] = ()


@functools.cache
def _language_models() -> tuple[tuple[Path, str], ...]:
    try:
        from docfgp_languages import REVISION, TESSDATA
    except ImportError:
        return ()
    return ((Path(TESSDATA), str(REVISION)),)


def _holds(directory: Path, names: list[str]) -> bool:
    return all((directory / f"{name}.traineddata").is_file() for name in names)


def _directories() -> tuple[tuple[Path, str], ...]:
    return ((_TESSDATA, _MANIFEST["revision"]), *_language_models())


def _holders(model: str) -> list[tuple[Path, str]]:
    return [found for found in _directories() if _holds(found[0], [model])]


def _resolve(config: FingerprintConfig) -> tuple[Path, str, str, str]:
    names = config.ocr_language.split("+")
    if config.tessdata_path is not None:
        custom = Path(config.tessdata_path).expanduser().resolve()
        if not _holds(custom, names):
            raise FingerprintError(f"tessdata_path lacks models for {config.ocr_language}")
        return custom, config.ocr_language, config.ocr_language, "custom"
    holders = {}
    for name in names:
        model = "script/Latin" if name in _LATIN_LANGUAGES and not _holders(name) else name
        holders[model] = _holders(model)
    missing = "+".join(model for model, found in holders.items() if not found)
    if missing:
        if _language_models():
            raise FingerprintError(f"No installed OCR models for {missing}; set tessdata_path")
        raise FingerprintError(
            f'No installed OCR models for {missing}; install "docfgp[languages]" '
            "for non-Latin scripts or set tessdata_path"
        )
    shared = [found for found in _directories() if all(found in h for h in holders.values())]
    chosen = {model: shared[0] if shared else found[0] for model, found in holders.items()}
    unloadable = f"OCR models for {config.ocr_language} cannot load together; set tessdata_path"
    try:
        root = Path(os.path.commonpath([directory for directory, _ in chosen.values()]))
    except ValueError as exc:
        raise FingerprintError(unloadable) from exc
    paths = [
        (directory / model).relative_to(root).as_posix() for model, (directory, _) in chosen.items()
    ]
    if any("+" in path or path.startswith("~") for path in paths):
        raise FingerprintError(unloadable)
    revisions = "+".join(dict.fromkeys(revision for _, revision in chosen.values()))
    return root, "+".join(paths), "+".join(chosen), f"tessdata_fast@{revisions}"


def model_directory(config: FingerprintConfig) -> Path:
    """Locate the Tesseract data directory that the configured OCR models load from.

    Args:
        config: OCR language and optional model directory.

    Returns:
        The configured, bundled, or docfgp-languages model directory, or the parent of both.

    Raises:
        FingerprintError: A configured model is missing or cannot load with the others.
    """
    return _resolve(config)[0]


def provenance(config: FingerprintConfig) -> dict[str, str]:
    """Identify the OCR engine and models that produce recognized text.

    Args:
        config: OCR language and optional model directory.

    Returns:
        Model languages, Tesseract version, and the pinned model revision, or ``custom``.

    Raises:
        FingerprintError: A configured model is missing or cannot load with the others.
    """
    _, _, language, models = _resolve(config)
    return {
        "ocr_language": language,
        "ocr_engine": f"tesseract-{tesserocr.tesseract_version().split()[1]}",
        "ocr_models": models,
    }


class Reader:
    """Recognize the pages of one document, loading the OCR models on first use.

    Args:
        config: OCR models, language, and timeout settings.
    """

    def __init__(self, config: FingerprintConfig) -> None:
        """Store the settings without loading models.

        Args:
            config: OCR models, language, and timeout settings.
        """
        self.config = config
        self._api: tesserocr.PyTessBaseAPI | None = None

    def read_text(self, image: Image.Image, deadline: float) -> OCRResult:
        """Extract text from a rendered page with the bundled Tesseract library.

        Args:
            image: Page image from `normalized_grayscale` within configured pixel limits.
            deadline: Cooperative document deadline.

        Returns:
            Text, words, and Tesseract word confidence from the same OCR invocation.

        Raises:
            FingerprintError: Tesseract models are missing, or Tesseract times out or fails.
        """
        timeout = min(self.config.process_timeout, remaining_seconds(deadline))
        try:
            if self._api is None:
                directory, models, _, _ = _resolve(self.config)
                self._api = tesserocr.PyTessBaseAPI(
                    path=f"{directory}{os.sep}",
                    lang=models,
                    psm=tesserocr.PSM.AUTO,
                    variables={"debug_file": os.devnull},
                )
            self._api.SetImage(image)
            completed = self._api.Recognize(timeout=math.ceil(timeout * 1000))
            remaining_seconds(deadline)
            if not completed:
                raise FingerprintError("Tesseract could not complete within process_timeout")
            text = self._api.GetUTF8Text()
            confidence_data = self._api.GetTSVText(0)
        except RuntimeError as exc:
            raise FingerprintError(f"Tesseract failed: {exc}") from exc
        if len(text.encode()) > self.config.max_text_chars * 4:
            raise FingerprintError("Tesseract produced oversized text")
        if len(confidence_data.encode()) > self.config.max_text_chars * 16:
            raise FingerprintError("Tesseract produced oversized confidence data")
        words = []
        try:
            for row in csv.DictReader(
                io.StringIO(confidence_data, newline=""), _TSV_FIELDS, delimiter="\t"
            ):
                if row["level"] != "5" or not row["text"].strip():
                    continue
                confidence = float(row["conf"])
                if math.isfinite(confidence) and confidence >= 0:
                    left, top = int(row["left"]), int(row["top"])
                    right, bottom = left + int(row["width"]), top + int(row["height"])
                    line = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
                    bounds = (left, top, right, bottom)
                    words.append(OCRWord(row["text"], confidence, bounds, line))
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise FingerprintError(f"Cannot read Tesseract confidence data: {exc}") from exc
        return OCRResult(text, mean_confidence(words) if words else None, tuple(words))

    def close(self) -> None:
        """Release the loaded OCR models."""
        if self._api is not None:
            self._api.End()
            self._api = None


def mean_confidence(words: Sequence[OCRWord]) -> float:
    """Average word confidence.

    Args:
        words: At least one recognized word.

    Returns:
        Mean confidence rounded to two decimals.
    """
    return round(sum(word.confidence for word in words) / len(words), 2)
