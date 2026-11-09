"""Exercise PDF, image, OCR, and CLI output contracts."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import reportlab
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas

from docfgp import DocumentFingerprint, FingerprintConfig, Fingerprinter, FingerprintError, ocr


def _pdf(path: Path, text: str) -> None:
    document = canvas.Canvas(str(path))
    document.drawString(72, 720, text)
    document.save()


def _scan_image(path: Path) -> None:
    from PIL import ImageFont

    image = Image.new("RGB", (1000, 220), "white")
    font = ImageFont.truetype(str(Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"), 36)
    ImageDraw.Draw(image).text(
        (30, 60), "Invoice total 12345 alpha beta gamma delta epsilon", font=font, fill="black"
    )
    image.save(path)


def test_pdf_is_deterministic_and_serializable(tmp_path: Path) -> None:
    """Generate identical complete fingerprints from repeat PDF invocations."""
    path = tmp_path / "one.pdf"
    _pdf(path, "alpha beta gamma delta epsilon zeta 123")
    fingerprinter = Fingerprinter(FingerprintConfig(tesseract_path="missing-tesseract"))
    first = fingerprinter.fingerprint(path)
    second = fingerprinter.fingerprint(path)
    assert isinstance(first, DocumentFingerprint)
    assert first.to_json() == second.to_json()
    assert first.quality.status == "text_and_visual"
    assert first.page_count == 1
    assert first.shingle_total == 3
    assert first.exact_hashes["canonical_sha256"]
    assert len(first.minhash) == 512
    assert len(first.lookup_keys["word_lsh"]) == 32
    assert len(first.page_hashes) == 1
    assert json.loads(first.to_json()) == first.to_dict()
    assert first.quality.text_evidence == "word_shingles"
    with pytest.raises(TypeError):
        first.exact_hashes["raw_sha256"] = "changed"
    data = first.to_dict()
    data["exact_hashes"]["raw_sha256"] = "changed"
    assert first.exact_hashes["raw_sha256"] != "changed"


def test_image_and_blank_evidence_states(tmp_path: Path) -> None:
    """Report visual-only and insufficient-evidence outcomes explicitly."""
    path = tmp_path / "image.png"
    image = Image.new("RGB", (300, 100), "white")
    ImageDraw.Draw(image).rectangle((30, 20, 100, 80), fill="black")
    image.save(path)
    config = FingerprintConfig(tesseract_path="missing-tesseract")
    visual = Fingerprinter(config).fingerprint(path)
    assert visual.quality.status == "visual_only"
    assert visual.exact_hashes["canonical_sha256"] is None
    assert visual.quality.pages[0].ocr_attempted
    assert visual.quality.warnings
    blank = tmp_path / "blank.png"
    Image.new("RGB", (100, 100), "white").save(blank)
    empty = Fingerprinter(config).fingerprint(blank)
    assert empty.quality.status == "insufficient_evidence"
    assert not empty.lookup_keys["visual"]


def test_image_metadata_and_transparency(tmp_path: Path) -> None:
    """Identify image content consistently and ignore fully transparent pixels."""
    image = Image.new("RGB", (80, 80), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 40, 40), fill="black")
    path = tmp_path / "photo.jpg"
    image.save(path)
    bundle = Fingerprinter(FingerprintConfig(tesseract_path="missing-tesseract")).fingerprint(path)
    assert bundle.source_format == "jpeg"

    transparent = tmp_path / "transparent.png"
    Image.new("RGBA", (80, 80), (0, 0, 0, 0)).save(transparent)
    bundle = Fingerprinter(FingerprintConfig(tesseract_path="missing-tesseract")).fingerprint(
        transparent
    )
    assert bundle.quality.status == "insufficient_evidence"
    assert not bundle.page_hashes


def test_short_native_text_does_not_trigger_ocr(tmp_path: Path) -> None:
    """Keep valid short text without using text length as an OCR trigger."""
    path = tmp_path / "short.pdf"
    _pdf(path, "Short")
    bundle = Fingerprinter(FingerprintConfig(tesseract_path="missing-tesseract")).fingerprint(path)
    assert bundle.quality.pages[0].source == "native"
    assert not bundle.quality.pages[0].ocr_attempted
    assert bundle.quality.text_evidence == "exact_only"


def test_low_confidence_ocr_is_not_text_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep uncertain OCR out of exact hashes and text lookup keys."""
    path = tmp_path / "blank.pdf"
    _pdf(path, "")
    monkeypatch.setattr(ocr, "read_text", lambda *_args: ocr.OCRResult("imagined words", 5.0))
    bundle = Fingerprinter().fingerprint(path)
    assert bundle.quality.text_evidence == "none"
    assert bundle.exact_hashes["canonical_sha256"] is None
    assert bundle.quality.pages[0].warning == "OCR confidence was too low to use its text"


def test_limits_and_invalid_input(tmp_path: Path) -> None:
    """Reject missing, malformed, and oversized input."""
    missing = tmp_path / "missing.pdf"
    with pytest.raises(FingerprintError):
        Fingerprinter().fingerprint(missing)
    invalid = tmp_path / "invalid.pdf"
    invalid.write_bytes(b"not a pdf")
    with pytest.raises(FingerprintError):
        Fingerprinter().fingerprint(invalid)
    image = tmp_path / "large.png"
    Image.new("RGB", (101, 101)).save(image)
    with pytest.raises(FingerprintError, match="max_page_pixels"):
        Fingerprinter(FingerprintConfig(max_page_pixels=10_000)).fingerprint(image)
    document = tmp_path / "features.pdf"
    _pdf(document, "alpha beta gamma delta epsilon zeta eta")
    with pytest.raises(FingerprintError, match="max_features"):
        Fingerprinter(FingerprintConfig(max_features=2)).fingerprint(document)
    office = tmp_path / "broken.docx"
    with zipfile.ZipFile(office, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
    with pytest.raises(FingerprintError, match="required parts"):
        Fingerprinter().fingerprint(office)


def test_multipage_image_respects_frame_limit(tmp_path: Path) -> None:
    """Fingerprint every TIFF frame and enforce the page limit."""
    path = tmp_path / "pages.tiff"
    first = Image.new("RGB", (80, 80), "white")
    second = Image.new("RGB", (80, 80), "black")
    first.save(path, save_all=True, append_images=[second])
    config = FingerprintConfig(tesseract_path="missing-tesseract")
    assert Fingerprinter(config).fingerprint(path).page_count == 2
    with pytest.raises(FingerprintError, match="max_pages"):
        Fingerprinter(FingerprintConfig(max_pages=1)).fingerprint(path)


def test_cli_matches_python(tmp_path: Path) -> None:
    """Keep the CLI and Python API on one serialized schema."""
    path = tmp_path / "one.pdf"
    _pdf(path, "alpha beta gamma delta epsilon zeta 123")
    expected = Fingerprinter().fingerprint(path).to_dict()
    completed = subprocess.run(
        [sys.executable, "-m", "docfgp.cli", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == expected
    failure = subprocess.run(
        [sys.executable, "-m", "docfgp.cli", str(path.with_name("missing.pdf"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failure.returncode == 2
    assert not failure.stdout
    assert "Cannot read document" in failure.stderr


@pytest.mark.skipif(not os.environ.get("DOCFGP_TEST_TESSERACT"), reason="Tesseract not configured")
def test_scanned_pdf_uses_tesseract(tmp_path: Path) -> None:
    """Recover searchable words from a PDF containing only a page image."""
    image_path = tmp_path / "scan.png"
    _scan_image(image_path)
    pdf_path = tmp_path / "scan.pdf"
    document = canvas.Canvas(str(pdf_path), pagesize=(600, 250))
    document.drawImage(str(image_path), 0, 50, width=600, height=132)
    document.save()
    config = FingerprintConfig(tesseract_path=os.environ["DOCFGP_TEST_TESSERACT"])
    bundle = Fingerprinter(config).fingerprint(pdf_path)
    assert bundle.quality.pages[0].source == "ocr"
    assert bundle.quality.pages[0].ocr_confidence is not None
    assert bundle.shingle_total > 0


@pytest.mark.skipif(not os.environ.get("DOCFGP_TEST_TESSERACT"), reason="Tesseract not configured")
def test_rasterized_page_with_native_header_uses_ocr(tmp_path: Path) -> None:
    """Recognize substantial image text beneath a valid native header."""
    image_path = tmp_path / "body.png"
    _scan_image(image_path)
    path = tmp_path / "mixed.pdf"
    document = canvas.Canvas(str(path), pagesize=(600, 250))
    document.drawImage(str(image_path), 0, 50, width=600, height=132)
    document.drawString(20, 230, "Header")
    document.save()
    bundle = Fingerprinter(
        FingerprintConfig(tesseract_path=os.environ["DOCFGP_TEST_TESSERACT"])
    ).fingerprint(path)
    assert bundle.quality.pages[0].ocr_reason == "rasterized_content"
    assert bundle.quality.pages[0].source == "ocr"
    assert bundle.shingle_total > 0
