"""Run bounded Tesseract OCR and retain word confidence evidence."""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .schemas import FingerprintConfig, FingerprintError, _remaining_seconds
from .visual import normalized_grayscale


@dataclass(frozen=True)
class OCRResult:
    """Hold recognized text and mean confidence over recognized words."""

    text: str
    confidence: float | None


def read_text(image: Image.Image, config: FingerprintConfig, deadline: float) -> OCRResult:
    """Extract text from a rendered page in an isolated temporary directory.

    Args:
        image: Page image within configured pixel limits.
        config: OCR executable, language, and timeout settings.
        deadline: Cooperative document deadline.

    Returns:
        Text and Tesseract word confidence from the same OCR invocation.

    Raises:
        FingerprintError: Tesseract is missing, times out, or fails.
    """
    configured = config.tesseract_path
    executable = (
        str(Path(configured).expanduser().resolve())
        if configured and Path(configured).expanduser().is_file()
        else shutil.which(configured or "tesseract")
    )
    if not executable:
        raise FingerprintError("Tesseract 5 is not available")
    with tempfile.TemporaryDirectory(prefix="docfgp-ocr-") as directory:
        input_path = Path(directory) / "page.png"
        output_base = Path(directory) / "result"
        output_path = output_base.with_suffix(".txt")
        confidence_path = output_base.with_suffix(".tsv")
        error_path = Path(directory) / "ocr.stderr"
        with normalized_grayscale(image) as prepared:
            prepared.save(input_path)
        command = [
            executable,
            str(input_path),
            str(output_base),
            "-l",
            config.ocr_language,
            "--psm",
            "3",
            "txt",
            "tsv",
        ]
        try:
            with error_path.open("wb") as errors:
                result = subprocess.run(
                    command,
                    cwd=directory,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                    timeout=min(config.process_timeout, _remaining_seconds(deadline)),
                    check=False,
                    env={**os.environ, "OMP_THREAD_LIMIT": "1"},
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _remaining_seconds(deadline)
            raise FingerprintError(f"Tesseract could not complete: {exc}") from exc
        _remaining_seconds(deadline)
        if result.returncode:
            with error_path.open("rb") as errors:
                message = errors.read(500).decode("utf-8", "replace").strip()
            raise FingerprintError(f"Tesseract failed: {message}")
        if not output_path.is_file() or output_path.stat().st_size > config.max_text_chars * 4:
            raise FingerprintError("Tesseract produced missing or oversized text")
        if (
            not confidence_path.is_file()
            or confidence_path.stat().st_size > config.max_text_chars * 16
        ):
            raise FingerprintError("Tesseract produced missing or oversized confidence data")
        total_confidence = 0.0
        word_count = 0
        try:
            with confidence_path.open(encoding="utf-8", errors="replace", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    if row["level"] != "5" or not row["text"].strip():
                        continue
                    confidence = float(row["conf"])
                    if math.isfinite(confidence) and confidence >= 0:
                        total_confidence += confidence
                        word_count += 1
        except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
            raise FingerprintError(f"Cannot read Tesseract confidence data: {exc}") from exc
        return OCRResult(
            output_path.read_text(encoding="utf-8", errors="replace"),
            round(total_confidence / word_count, 2) if word_count else None,
        )
