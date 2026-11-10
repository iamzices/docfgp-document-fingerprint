"""Run bounded Tesseract OCR and retain word confidence evidence."""

from __future__ import annotations

import csv
import functools
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import tesserocr
from PIL import Image

from .schemas import FingerprintConfig, FingerprintError, _remaining_seconds
from .visual import normalized_grayscale

_TESSDATA = Path(__file__).with_name("tessdata")
_MANIFEST = json.loads((_TESSDATA / "models.json").read_text(encoding="utf-8"))
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
class OCRResult:
    """Hold recognized text and mean confidence over recognized words."""

    text: str
    confidence: float | None


@functools.cache
def _language_models() -> tuple[tuple[Path, str], ...]:
    try:
        from docfgp_languages import REVISION, TESSDATA
    except ImportError:
        return ()
    return ((Path(TESSDATA), str(REVISION)),)


def _holds(directory: Path, names: list[str]) -> bool:
    return all((directory / f"{name}.traineddata").is_file() for name in names)


def _resolve(config: FingerprintConfig) -> tuple[Path, str]:
    names = config.ocr_language.split("+")
    if config.tessdata_path is not None:
        custom = Path(config.tessdata_path).expanduser().resolve()
        if not _holds(custom, names):
            raise FingerprintError(f"tessdata_path lacks models for {config.ocr_language}")
        return custom, "custom"
    for directory, revision in ((_TESSDATA, _MANIFEST["revision"]), *_language_models()):
        if _holds(directory, names):
            return directory, f"tessdata_fast@{revision}"
    if _language_models():
        raise FingerprintError(
            f"No installed OCR models for {config.ocr_language}; set tessdata_path"
        )
    raise FingerprintError(
        f'OCR models for {config.ocr_language} require "docfgp[languages]" or tessdata_path'
    )


def model_directory(config: FingerprintConfig) -> Path:
    """Locate one Tesseract data directory holding every configured OCR model.

    Args:
        config: OCR language and optional model directory.

    Returns:
        The configured, bundled, or docfgp-languages model directory.

    Raises:
        FingerprintError: No available directory holds every configured model.
    """
    return _resolve(config)[0]


def provenance(config: FingerprintConfig) -> dict[str, str]:
    """Identify the OCR engine and models that produce recognized text.

    Args:
        config: OCR language and optional model directory.

    Returns:
        Tesseract version and the pinned model revision, or ``custom``.

    Raises:
        FingerprintError: No available directory holds every configured model.
    """
    return {
        "ocr_engine": f"tesseract-{tesserocr.tesseract_version().split()[1]}",
        "ocr_models": _resolve(config)[1],
    }


def read_text(image: Image.Image, config: FingerprintConfig, deadline: float) -> OCRResult:
    """Extract text from a rendered page with the bundled Tesseract library.

    Args:
        image: Page image within configured pixel limits.
        config: OCR models, language, and timeout settings.
        deadline: Cooperative document deadline.

    Returns:
        Text and Tesseract word confidence from the same OCR invocation.

    Raises:
        FingerprintError: Tesseract models are missing, or Tesseract times out or fails.
    """
    timeout = min(config.process_timeout, _remaining_seconds(deadline))
    try:
        with (
            tesserocr.PyTessBaseAPI(
                path=f"{model_directory(config)}{os.sep}",
                lang=config.ocr_language,
                psm=tesserocr.PSM.AUTO,
                variables={"debug_file": os.devnull},
            ) as api,
            normalized_grayscale(image) as prepared,
        ):
            api.SetImage(prepared)
            completed = api.Recognize(timeout=math.ceil(timeout * 1000))
            _remaining_seconds(deadline)
            if not completed:
                raise FingerprintError("Tesseract could not complete within process_timeout")
            text = api.GetUTF8Text()
            confidence_data = api.GetTSVText(0)
    except RuntimeError as exc:
        raise FingerprintError(f"Tesseract failed: {exc}") from exc
    if len(text.encode()) > config.max_text_chars * 4:
        raise FingerprintError("Tesseract produced oversized text")
    if len(confidence_data.encode()) > config.max_text_chars * 16:
        raise FingerprintError("Tesseract produced oversized confidence data")
    total_confidence = 0.0
    word_count = 0
    try:
        for row in csv.DictReader(
            io.StringIO(confidence_data, newline=""), _TSV_FIELDS, delimiter="\t"
        ):
            if row["level"] != "5" or not row["text"].strip():
                continue
            confidence = float(row["conf"])
            if math.isfinite(confidence) and confidence >= 0:
                total_confidence += confidence
                word_count += 1
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise FingerprintError(f"Cannot read Tesseract confidence data: {exc}") from exc
    return OCRResult(text, round(total_confidence / word_count, 2) if word_count else None)
