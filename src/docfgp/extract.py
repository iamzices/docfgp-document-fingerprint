"""Validate input and extract PDF or image pages with bounded decoding."""

from __future__ import annotations

import contextlib
import math
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

from .normalize import strict_text, word_tokens
from .schemas import FingerprintConfig, FingerprintError, OCRReason, _remaining_seconds

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_IMAGE_FORMATS = {"PNG", "JPEG", "TIFF", "BMP", "WEBP"}
_OFFICE_SUFFIXES = {".docx", ".pptx"}
RASTER_OCR_COVERAGE = 0.5


@dataclass(frozen=True)
class ExtractedPage:
    """Carry one page's native text, image renders, and OCR trigger."""

    number: int
    native_text: str
    visual: Image.Image
    ocr_image: Image.Image | None
    ocr_reason: OCRReason | None


def identify(path: Path, config: FingerprintConfig) -> tuple[str, int]:
    """Validate the format signature, size, and Office ZIP structure.

    Args:
        path: Existing input path.
        config: Input and expansion limits.

    Returns:
        Format name and original byte size.

    Raises:
        FingerprintError: The file is unsupported, unsafe, or invalid.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(16)
    except OSError as exc:
        raise FingerprintError(f"Cannot read document: {exc}") from exc
    if size == 0 or size > config.max_file_bytes:
        raise FingerprintError("Document is empty or exceeds max_file_bytes")
    suffix = path.suffix.lower()
    if suffix == ".pdf" and header.startswith(b"%PDF-"):
        return "pdf", size
    if suffix in _OFFICE_SUFFIXES and header.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > 10_000 or sum(item.file_size for item in infos) > 200_000_000:
                    raise FingerprintError("Office package exceeds expansion limits")
                names = set(archive.namelist())
                required = "word/document.xml" if suffix == ".docx" else "ppt/presentation.xml"
                if "[Content_Types].xml" not in names or required not in names:
                    raise FingerprintError("Office package is missing required parts")
                if any(item.flag_bits & 1 for item in infos):
                    raise FingerprintError("Encrypted Office packages are unsupported")
                if archive.testzip() is not None:
                    raise FingerprintError("Office package failed CRC validation")
        except (OSError, zipfile.BadZipFile, RuntimeError, ValueError, EOFError) as exc:
            raise FingerprintError(f"Invalid Office package: {exc}") from exc
        return suffix[1:], size
    if suffix in _IMAGE_SUFFIXES:
        try:
            with Image.open(path) as image:
                image_format = image.format
                if image_format not in _IMAGE_FORMATS:
                    raise FingerprintError("Unsupported image format")
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise FingerprintError(f"Invalid image: {exc}") from exc
        return image_format.lower(), size
    raise FingerprintError("Unsupported format or file signature")


@contextlib.contextmanager
def pdf_path(
    path: Path, source_format: str, config: FingerprintConfig, deadline: float
) -> Iterator[Path]:
    """Yield a PDF path, converting Office files in a temporary directory.

    Args:
        path: Original path.
        source_format: Identified input format.
        config: Converter timeout and output limit.
        deadline: Cooperative document deadline.

    Yields:
        Original or temporary PDF path.

    Raises:
        FingerprintError: Conversion fails or produces invalid output.
    """
    if source_format == "pdf":
        _remaining_seconds(deadline)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="docfgp-office-") as directory:
        output = Path(directory) / "converted.pdf"
        error_path = Path(directory) / "converter.stderr"
        try:
            with error_path.open("wb") as errors:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-P",
                        "-m",
                        "docfgp._office",
                        str(path.resolve()),
                        str(output),
                    ],
                    cwd=directory,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                    timeout=min(config.process_timeout, _remaining_seconds(deadline)),
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _remaining_seconds(deadline)
            raise FingerprintError(f"office2pdf could not complete: {exc}") from exc
        _remaining_seconds(deadline)
        if result.returncode or not output.is_file():
            with error_path.open("rb") as errors:
                message = errors.read(500).decode("utf-8", "replace").strip()
            raise FingerprintError(f"office2pdf failed: {message}")
        with output.open("rb") as stream:
            signature = stream.read(5)
        if output.stat().st_size > config.max_file_bytes or signature != b"%PDF-":
            raise FingerprintError("office2pdf produced an invalid or oversized PDF")
        yield output


def _render(page: pdfium.PdfPage, dpi: int, config: FingerprintConfig) -> Image.Image:
    width, height = page.get_size()
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise FingerprintError("Invalid PDF page geometry")
    scale = min(dpi / 72, math.sqrt(config.max_page_pixels / (width * height)))
    if scale <= 0:
        raise FingerprintError("PDF page exceeds rendering limits")
    image = page.render(scale=scale).to_pil()
    if image.width * image.height > config.max_page_pixels:
        image.close()
        raise FingerprintError("PDF page exceeds max_page_pixels")
    return image


def _ocr_reason(page: pdfium.PdfPage, text: str) -> OCRReason | None:
    if any(
        character == "\ufffd"
        or (unicodedata.category(character) == "Cc" and character not in "\r\n\t\f")
        for character in text
    ):
        return "invalid_native_text"
    if not word_tokens(strict_text(text)):
        return "missing_native_text"
    width, height = page.get_size()
    page_area = width * height
    if not math.isfinite(page_area) or page_area <= 0:
        raise FingerprintError("Invalid PDF page geometry")
    image_area = 0.0
    for image in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
        left, bottom, right, top = image.get_bounds()
        if not all(math.isfinite(value) for value in (left, bottom, right, top)):
            continue
        visible_width = max(0.0, min(right, width) - max(left, 0.0))
        visible_height = max(0.0, min(top, height) - max(bottom, 0.0))
        image_area += visible_width * visible_height
        if image_area >= page_area * RASTER_OCR_COVERAGE:
            return "rasterized_content"
    return None


def pdf_pages(
    path: Path, config: FingerprintConfig, deadline: float
) -> Generator[ExtractedPage, None, None]:
    """Yield native text plus visual and optional OCR render for each PDF page.

    Args:
        path: Input PDF.
        config: Page, pixel, and resolution limits.
        deadline: Cooperative document deadline.

    Yields:
        Page data with native text and bounded image renders.

    Raises:
        FingerprintError: PDF opening, page decoding, or limits fail.
    """
    try:
        document = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise FingerprintError(f"Cannot open PDF: {exc}") from exc
    try:
        count = len(document)
        if count == 0 or count > config.max_pages:
            raise FingerprintError("PDF has no pages or exceeds max_pages")
        for index in range(count):
            _remaining_seconds(deadline)
            try:
                page = document[index]
            except Exception as exc:
                raise FingerprintError(f"Cannot open PDF page {index + 1}: {exc}") from exc
            try:
                textpage = page.get_textpage()
                try:
                    # pdfium replaces a hyphen at a line break with U+FFFE.
                    text = textpage.get_text_range().replace("\ufffe", "")
                finally:
                    textpage.close()
                reason = _ocr_reason(page, text)
                visual = _render(page, config.render_dpi, config)
                try:
                    ocr_image = _render(page, config.ocr_dpi, config) if reason else None
                except Exception:
                    visual.close()
                    raise
                yield ExtractedPage(index + 1, text, visual, ocr_image, reason)
            except FingerprintError:
                raise
            except Exception as exc:
                raise FingerprintError(f"Cannot process PDF page {index + 1}: {exc}") from exc
            finally:
                page.close()
    finally:
        document.close()


def image_pages(
    path: Path, config: FingerprintConfig, deadline: float
) -> Generator[ExtractedPage, None, None]:
    """Yield decoded image frames within configured limits.

    Args:
        path: Standalone image path.
        config: Frame and pixel limits.
        deadline: Cooperative document deadline.

    Yields:
        One-based image frames with OCR provenance.

    Raises:
        FingerprintError: Decoding fails or limits are exceeded.
    """
    try:
        with Image.open(path) as image:
            frames = getattr(image, "n_frames", 1)
            if frames > config.max_pages:
                raise FingerprintError("Image exceeds max_pages")
            for index in range(frames):
                _remaining_seconds(deadline)
                image.seek(index)
                if image.width * image.height > config.max_page_pixels:
                    raise FingerprintError("Image frame exceeds max_page_pixels")
                frame = image.copy()
                yield ExtractedPage(index + 1, "", frame, frame, "image_input")
    except (OSError, UnidentifiedImageError) as exc:
        raise FingerprintError(f"Cannot decode image: {exc}") from exc
