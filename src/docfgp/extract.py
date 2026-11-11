"""Validate input and extract PDF or image pages with bounded decoding."""

from __future__ import annotations

import contextlib
import ctypes
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
from PIL import Image

from .normalize import strict_text, word_tokens
from .schemas import Box, Deadline, FingerprintConfig, FingerprintError, OCRReason, converted_errors
from .visual import normalized_grayscale

RASTER_OCR_COVERAGE = 0.5
_HEADER_BYTES = 16
_PDF_SIGNATURE = b"%PDF-"
_ZIP_SIGNATURE = b"PK\x03\x04"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_IMAGE_FORMATS = {"PNG", "JPEG", "TIFF", "BMP", "WEBP"}
_OFFICE_MAIN_PARTS = {".docx": "word/document.xml", ".pptx": "ppt/presentation.xml"}
_OFFICE_FORMATS = {"docx", "pptx"}
_MAX_ZIP_ENTRIES = 10_000
_MAX_ZIP_EXPANDED_BYTES = 200_000_000
_ZIP_ENCRYPTED_FLAG = 0x1
_CONVERTER_MESSAGE_BYTES = 500
_POINTS_PER_INCH = 72
_LAYOUT_CONTROL_CHARACTERS = "\r\n\t\f"


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Carry one page's native text, grayscale renders, OCR trigger, and native text boxes."""

    number: int
    native_text: str
    visual: Image.Image
    ocr_image: Image.Image | None
    ocr_reason: OCRReason | None
    native_text_boxes: tuple[Box, ...] = ()

    def close(self) -> None:
        """Release the page renders."""
        self.visual.close()
        if self.ocr_image is not None and self.ocr_image is not self.visual:
            self.ocr_image.close()


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
    header, size = _read_header(path)
    if size == 0 or size > config.max_file_bytes:
        raise FingerprintError("Document is empty or exceeds max_file_bytes")

    suffix = path.suffix.lower()
    if suffix == ".pdf" and header.startswith(_PDF_SIGNATURE):
        return "pdf", size
    if suffix in _OFFICE_MAIN_PARTS and header.startswith(_ZIP_SIGNATURE):
        _check_office_package(path, _OFFICE_MAIN_PARTS[suffix])
        return suffix.removeprefix("."), size
    if suffix in _IMAGE_SUFFIXES:
        return _check_image(path), size
    raise FingerprintError("Unsupported format or file signature")


@contextlib.contextmanager
def pages(
    path: Path, source_format: str, config: FingerprintConfig, deadline: Deadline
) -> Iterator[Generator[ExtractedPage, None, None]]:
    """Yield an identified document's pages and release them afterwards.

    Args:
        path: Original path.
        source_format: Format returned by `identify`.
        config: Conversion, page, and pixel limits.
        deadline: Cooperative document deadline.

    Yields:
        Extracted pages in document order.

    Raises:
        FingerprintError: Conversion, decoding, or limits fail.
    """
    if source_format == "pdf":
        with contextlib.closing(_pdf_pages(path, config, deadline)) as pdf_pages:
            yield pdf_pages
    elif source_format in _OFFICE_FORMATS:
        with (
            _converted_pdf(path, config, deadline) as pdf,
            contextlib.closing(_pdf_pages(pdf, config, deadline)) as pdf_pages,
        ):
            yield pdf_pages
    else:
        with contextlib.closing(_image_pages(path, config, deadline)) as image_pages:
            yield image_pages


def _read_header(path: Path) -> tuple[bytes, int]:
    with converted_errors("Cannot read document"):
        size = path.stat().st_size
        with path.open("rb") as stream:
            return stream.read(_HEADER_BYTES), size


def _check_office_package(path: Path, main_part: str) -> None:
    with converted_errors("Invalid Office package"), zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        expanded_bytes = sum(entry.file_size for entry in entries)
        if len(entries) > _MAX_ZIP_ENTRIES or expanded_bytes > _MAX_ZIP_EXPANDED_BYTES:
            raise FingerprintError("Office package exceeds expansion limits")
        names = {entry.filename for entry in entries}
        if "[Content_Types].xml" not in names or main_part not in names:
            raise FingerprintError("Office package is missing required parts")
        if any(entry.flag_bits & _ZIP_ENCRYPTED_FLAG for entry in entries):
            raise FingerprintError("Encrypted Office packages are unsupported")
        if archive.testzip() is not None:
            raise FingerprintError("Office package failed CRC validation")


def _check_image(path: Path) -> str:
    with converted_errors("Invalid image"), Image.open(path) as image:
        image_format = image.format
        if image_format not in _IMAGE_FORMATS:
            raise FingerprintError("Unsupported image format")
        image.verify()
    return image_format.lower()


@contextlib.contextmanager
def _converted_pdf(path: Path, config: FingerprintConfig, deadline: Deadline) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="docfgp-office-") as directory:
        pdf = Path(directory) / "converted.pdf"
        _run_office_converter(path, pdf, config, deadline)

        with pdf.open("rb") as stream:
            signature = stream.read(len(_PDF_SIGNATURE))
        if pdf.stat().st_size > config.max_file_bytes or signature != _PDF_SIGNATURE:
            raise FingerprintError("office2pdf produced an invalid or oversized PDF")

        yield pdf


def _run_office_converter(
    path: Path, pdf: Path, config: FingerprintConfig, deadline: Deadline
) -> None:
    messages = pdf.with_name("converter.stderr")
    command = [sys.executable, "-P", "-m", "docfgp._office", str(path.resolve()), str(pdf)]

    try:
        with messages.open("wb") as stderr:
            result = subprocess.run(
                command,
                cwd=pdf.parent,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                timeout=min(config.process_timeout, deadline.remaining()),
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # A timeout caused by the document deadline is reported as the deadline.
        deadline.check()
        raise FingerprintError(f"office2pdf could not complete: {exc}") from exc

    deadline.check()
    if result.returncode or not pdf.is_file():
        with messages.open("rb") as stderr:
            message = stderr.read(_CONVERTER_MESSAGE_BYTES).decode("utf-8", "replace").strip()
        raise FingerprintError(f"office2pdf failed: {message}")


def _pdf_pages(
    path: Path, config: FingerprintConfig, deadline: Deadline
) -> Generator[ExtractedPage, None, None]:
    with converted_errors("Cannot open PDF"):
        document = pdfium.PdfDocument(str(path))

    with contextlib.closing(document):
        page_count = len(document)
        if page_count == 0 or page_count > config.max_pages:
            raise FingerprintError("PDF has no pages or exceeds max_pages")

        for index in range(page_count):
            deadline.check()
            number = index + 1
            with converted_errors(f"Cannot open PDF page {number}"):
                page = document[index]
            with contextlib.closing(page), converted_errors(f"Cannot process PDF page {number}"):
                yield _extract_pdf_page(page, number, config)


def _extract_pdf_page(
    page: pdfium.PdfPage, number: int, config: FingerprintConfig
) -> ExtractedPage:
    with contextlib.closing(page.get_textpage()) as textpage:
        # pdfium replaces a hyphen at a line break with U+FFFE.
        text = textpage.get_text_range().replace("\ufffe", "")
        reason = _ocr_reason(page, text)
        text_rects = []
        if reason == "rasterized_content":
            text_rects = [textpage.get_rect(index) for index in range(textpage.count_rects())]

    visual = _render(page, config.render_dpi, config)
    try:
        ocr_image = _render(page, config.ocr_dpi, config) if reason else None
    except BaseException:
        visual.close()
        raise

    if ocr_image is None:
        return ExtractedPage(number, text, visual, None, reason)
    native_text_boxes = tuple(_device_box(page, rect, ocr_image.size) for rect in text_rects)
    return ExtractedPage(number, text, visual, ocr_image, reason, native_text_boxes)


def _render(page: pdfium.PdfPage, dpi: int, config: FingerprintConfig) -> Image.Image:
    width, height = page.get_size()
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise FingerprintError("Invalid PDF page geometry")

    # Largest scale whose ceil-rounded render still fits max_page_pixels.
    area, perimeter = width * height, width + height
    pixels = config.max_page_pixels - 1
    fitting_scale = (math.sqrt(perimeter**2 + 4 * area * pixels) - perimeter) / (2 * area)
    scale = min(dpi / _POINTS_PER_INCH, fitting_scale)
    if scale <= 0:
        raise FingerprintError("PDF page exceeds rendering limits")

    with page.render(scale=scale).to_pil() as image:
        if image.width * image.height > config.max_page_pixels:
            raise FingerprintError("PDF page exceeds max_page_pixels")
        return normalized_grayscale(image)


def _device_box(
    page: pdfium.PdfPage, rect: tuple[float, float, float, float], size: tuple[int, int]
) -> Box:
    left, bottom, right, top = rect
    x0, y0 = _device_point(page, left, bottom, size)
    x1, y1 = _device_point(page, right, top, size)
    return Box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _device_point(
    page: pdfium.PdfPage, x: float, y: float, size: tuple[int, int]
) -> tuple[int, int]:
    # pdfium applies the page's rotation and origin when mapping points to render pixels.
    device_x, device_y = ctypes.c_int(), ctypes.c_int()
    width, height = size
    if not pdfium.raw.FPDF_PageToDevice(page.raw, 0, 0, width, height, 0, x, y, device_x, device_y):
        raise FingerprintError("Cannot map PDF text to page pixels")
    return device_x.value, device_y.value


def _ocr_reason(page: pdfium.PdfPage, text: str) -> OCRReason | None:
    if _has_invalid_characters(text):
        return "invalid_native_text"
    if not word_tokens(strict_text(text)):
        return "missing_native_text"
    if _is_mostly_images(page):
        return "rasterized_content"
    return None


def _has_invalid_characters(text: str) -> bool:
    for character in text:
        if character == "\ufffd":
            return True
        if unicodedata.category(character) == "Cc" and character not in _LAYOUT_CONTROL_CHARACTERS:
            return True
    return False


def _is_mostly_images(page: pdfium.PdfPage) -> bool:
    width, height = page.get_size()
    page_area = width * height
    if not math.isfinite(page_area) or page_area <= 0:
        raise FingerprintError("Invalid PDF page geometry")

    image_area = 0.0
    for image in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
        image_area += _visible_area(image.get_bounds(), width, height)
        if image_area >= page_area * RASTER_OCR_COVERAGE:
            return True
    return False


def _visible_area(
    bounds: tuple[float, float, float, float], page_width: float, page_height: float
) -> float:
    left, bottom, right, top = bounds
    if not all(math.isfinite(edge) for edge in bounds):
        return 0.0

    visible_width = max(0.0, min(right, page_width) - max(left, 0.0))
    visible_height = max(0.0, min(top, page_height) - max(bottom, 0.0))
    return visible_width * visible_height


def _image_pages(
    path: Path, config: FingerprintConfig, deadline: Deadline
) -> Generator[ExtractedPage, None, None]:
    with converted_errors("Cannot decode image"), Image.open(path) as image:
        frame_count = getattr(image, "n_frames", 1)
        if frame_count > config.max_pages:
            raise FingerprintError("Image exceeds max_pages")

        for index in range(frame_count):
            deadline.check()
            image.seek(index)
            if image.width * image.height > config.max_image_pixels:
                raise FingerprintError("Image frame exceeds max_image_pixels")
            with image.copy() as frame:
                grayscale = normalized_grayscale(frame)
            yield ExtractedPage(index + 1, "", grayscale, grayscale, "image_input")
