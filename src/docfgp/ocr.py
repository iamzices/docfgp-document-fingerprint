"""Run bounded Tesseract OCR and retain word confidence evidence."""

from __future__ import annotations

import csv
import functools
import importlib
import io
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import tesserocr
from PIL import Image

from .schemas import Box, Deadline, FingerprintConfig, FingerprintError, converted_errors

_TESSDATA = Path(__file__).with_name("tessdata")
_MANIFEST = json.loads((_TESSDATA / "models.json").read_text(encoding="utf-8"))
_LATIN_SCRIPT_MODEL = "script/Latin"
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
_TSV_WORD_LEVEL = "5"
_MAX_UTF8_BYTES_PER_CHAR = 4
_MAX_TSV_BYTES_PER_CHAR = 16


@dataclass(frozen=True, slots=True)
class LinePosition:
    """Locate a recognized line by its Tesseract block, paragraph, and line numbers."""

    block: int
    paragraph: int
    line: int


@dataclass(frozen=True, slots=True)
class OCRWord:
    """Hold one recognized word with its confidence, pixel box, and line."""

    text: str
    confidence: float
    box: Box
    line: LinePosition


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Hold recognized text, its words, and mean confidence over recognized words."""

    text: str
    confidence: float | None
    words: tuple[OCRWord, ...] = ()


@dataclass(frozen=True, slots=True)
class _ModelDirectory:
    path: Path
    revision: str


@dataclass(frozen=True, slots=True)
class _ModelSelection:
    """Describe the models Tesseract loads.

    `tesseract_languages` holds model paths relative to `directory` in Tesseract's "+" syntax;
    `languages` holds the model names and `models` their provenance.
    """

    directory: Path
    tesseract_languages: str
    languages: str
    models: str


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
        self._config = config
        self._api: tesserocr.PyTessBaseAPI | None = None

    def read_text(self, image: Image.Image, deadline: Deadline) -> OCRResult:
        """Extract text from a rendered page with the bundled Tesseract library.

        Args:
            image: Page image from `normalized_grayscale` within configured pixel limits.
            deadline: Cooperative document deadline.

        Returns:
            Text, words, and Tesseract word confidence from the same OCR invocation.

        Raises:
            FingerprintError: Tesseract models are missing, or Tesseract times out or fails.
        """
        text, tsv = self._recognize(image, deadline)

        max_chars = self._config.max_text_chars
        if len(text.encode()) > max_chars * _MAX_UTF8_BYTES_PER_CHAR:
            raise FingerprintError("Tesseract produced oversized text")
        if len(tsv.encode()) > max_chars * _MAX_TSV_BYTES_PER_CHAR:
            raise FingerprintError("Tesseract produced oversized confidence data")

        words = _parse_words(tsv)
        return OCRResult(text, mean_confidence(words) if words else None, words)

    def close(self) -> None:
        """Release the loaded OCR models."""
        if self._api is not None:
            self._api.End()
            self._api = None

    def _recognize(self, image: Image.Image, deadline: Deadline) -> tuple[str, str]:
        timeout_ms = math.ceil(min(self._config.process_timeout, deadline.remaining()) * 1000)

        with converted_errors("Tesseract failed"):
            api = self._loaded_api()
            api.SetImage(image)
            completed = api.Recognize(timeout=timeout_ms)
            deadline.check()
            if not completed:
                raise FingerprintError("Tesseract could not complete within process_timeout")
            return api.GetUTF8Text(), api.GetTSVText(0)

    def _loaded_api(self) -> tesserocr.PyTessBaseAPI:
        if self._api is None:
            models = _find_models(self._config)
            self._api = tesserocr.PyTessBaseAPI(
                path=f"{models.directory}{os.sep}",
                lang=models.tesseract_languages,
                psm=tesserocr.PSM.AUTO,
                variables={"debug_file": os.devnull},
            )
        return self._api


def model_directory(config: FingerprintConfig) -> Path:
    """Locate the Tesseract data directory that the configured OCR models load from.

    Args:
        config: OCR language and optional model directory.

    Returns:
        The configured, bundled, or docfgp-languages model directory, or the parent of both.

    Raises:
        FingerprintError: A configured model is missing or cannot load with the others.
    """
    return _find_models(config).directory


def provenance(config: FingerprintConfig) -> dict[str, str]:
    """Identify the OCR engine and models that produce recognized text.

    Args:
        config: OCR language and optional model directory.

    Returns:
        Model languages, Tesseract version, and the pinned model revision, or ``custom``.

    Raises:
        FingerprintError: A configured model is missing or cannot load with the others.
    """
    models = _find_models(config)
    return {
        "ocr_language": models.languages,
        "ocr_engine": f"tesseract-{tesserocr.tesseract_version().split()[1]}",
        "ocr_models": models.models,
    }


def mean_confidence(words: Sequence[OCRWord]) -> float:
    """Average word confidence.

    Args:
        words: At least one recognized word.

    Returns:
        Mean confidence rounded to two decimals.
    """
    return round(sum(word.confidence for word in words) / len(words), 2)


def _find_models(config: FingerprintConfig) -> _ModelSelection:
    names = config.ocr_language.split("+")

    if config.tessdata_path is not None:
        directory = Path(config.tessdata_path).expanduser().resolve()
        if not _has_models(directory, names):
            raise FingerprintError(f"tessdata_path lacks models for {config.ocr_language}")
        return _ModelSelection(directory, config.ocr_language, config.ocr_language, "custom")

    holders = _model_holders(names)
    _require_installed(holders)
    return _select_models(holders, config.ocr_language)


def _model_holders(names: list[str]) -> dict[str, list[_ModelDirectory]]:
    holders = {}
    for name in names:
        model = name
        if name in _LATIN_LANGUAGES and not _directories_with(name):
            model = _LATIN_SCRIPT_MODEL
        holders[model] = _directories_with(model)
    return holders


def _require_installed(holders: dict[str, list[_ModelDirectory]]) -> None:
    missing = "+".join(model for model, directories in holders.items() if not directories)
    if not missing:
        return

    if _language_models():
        raise FingerprintError(f"No installed OCR models for {missing}; set tessdata_path")
    raise FingerprintError(
        f'No installed OCR models for {missing}; install "docfgp[languages]" '
        "for non-Latin scripts or set tessdata_path"
    )


def _select_models(
    holders: dict[str, list[_ModelDirectory]], language_codes: str
) -> _ModelSelection:
    shared = _directory_with_all(holders)
    chosen = {
        model: shared if shared is not None else directories[0]
        for model, directories in holders.items()
    }

    unloadable = f"OCR models for {language_codes} cannot load together; set tessdata_path"
    try:
        root = Path(os.path.commonpath([directory.path for directory in chosen.values()]))
    except ValueError as exc:
        raise FingerprintError(unloadable) from exc
    model_paths = [
        (directory.path / model).relative_to(root).as_posix() for model, directory in chosen.items()
    ]
    # Tesseract splits its language list on "+" and skips entries that start with "~".
    if any("+" in path or path.startswith("~") for path in model_paths):
        raise FingerprintError(unloadable)

    revisions = "+".join(dict.fromkeys(directory.revision for directory in chosen.values()))
    return _ModelSelection(
        directory=root,
        tesseract_languages="+".join(model_paths),
        languages="+".join(chosen),
        models=f"tessdata_fast@{revisions}",
    )


def _directory_with_all(holders: dict[str, list[_ModelDirectory]]) -> _ModelDirectory | None:
    for directory in _model_directories():
        if all(directory in directories for directories in holders.values()):
            return directory
    return None


def _directories_with(model: str) -> list[_ModelDirectory]:
    return [directory for directory in _model_directories() if _has_models(directory.path, [model])]


def _has_models(directory: Path, names: Sequence[str]) -> bool:
    return all((directory / f"{name}.traineddata").is_file() for name in names)


def _model_directories() -> tuple[_ModelDirectory, ...]:
    return (_ModelDirectory(_TESSDATA, _MANIFEST["revision"]), *_language_models())


@functools.cache
def _language_models() -> tuple[_ModelDirectory, ...]:
    try:
        languages = importlib.import_module("docfgp_languages")
    except ImportError:
        return ()
    return (_ModelDirectory(Path(languages.TESSDATA), str(languages.REVISION)),)


def _parse_words(tsv: str) -> tuple[OCRWord, ...]:
    rows = csv.DictReader(io.StringIO(tsv, newline=""), _TSV_FIELDS, delimiter="\t")
    with converted_errors("Cannot read Tesseract confidence data"):
        words = [_parse_word(row) for row in rows]
    return tuple(word for word in words if word is not None)


def _parse_word(row: dict[str, str]) -> OCRWord | None:
    if row["level"] != _TSV_WORD_LEVEL or not row["text"].strip():
        return None

    confidence = float(row["conf"])
    if not math.isfinite(confidence) or confidence < 0:
        return None

    left, top = int(row["left"]), int(row["top"])
    return OCRWord(
        text=row["text"],
        confidence=confidence,
        box=Box(left, top, left + int(row["width"]), top + int(row["height"])),
        line=LinePosition(int(row["block_num"]), int(row["par_num"]), int(row["line_num"])),
    )
