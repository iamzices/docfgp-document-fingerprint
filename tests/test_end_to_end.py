"""Exercise PDF, Office, image, OCR, and CLI output contracts."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import reportlab
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas

from docfgp import DocumentFingerprint, FingerprintConfig, Fingerprinter, FingerprintError, ocr
from docfgp.fingerprint import _ocr_addition, _ocr_replacement, _OCROutcome
from docfgp.schemas import Box


def _pdf(path: Path, text: str) -> None:
    document = canvas.Canvas(str(path))
    document.drawString(72, 720, text)
    document.save()


def _docx(path: Path, text: str) -> None:
    content_types = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
    )
    relationships = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="word/document.xml" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/officeDocument"/></Relationships>'
    )
    document = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)


def _languages_package(root: Path, source: str) -> dict[str, str]:
    package = root / "docfgp_languages"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source)

    path = os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH"))))
    return {**os.environ, "PYTHONPATH": path}


def _scan_image(path: Path) -> None:
    image = Image.new("RGB", (1000, 220), "white")
    font = ImageFont.truetype(str(Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"), 36)
    ImageDraw.Draw(image).text(
        (30, 60), "Invoice total 12345 alpha beta gamma delta epsilon", font=font, fill="black"
    )
    image.save(path)


def _recognized(text: str, confidence: float) -> ocr.OCRResult:
    words = tuple(
        ocr.OCRWord(
            word, confidence, Box(50 * index, 0, 50 * index + 40, 20), ocr.LinePosition(1, 1, 1)
        )
        for index, word in enumerate(text.split())
    )
    return ocr.OCRResult(text, confidence if words else None, words)


def test_pdf_is_deterministic_and_serializable(tmp_path: Path) -> None:
    """Generate identical complete fingerprints from repeat PDF invocations."""
    path = tmp_path / "one.pdf"
    _pdf(path, "alpha beta gamma delta epsilon zeta 123")

    fingerprinter = Fingerprinter()
    first = fingerprinter.fingerprint(path)
    second = fingerprinter.fingerprint(path)
    assert isinstance(first, DocumentFingerprint)
    assert first.to_json() == second.to_json()
    assert first.quality.status == "text_and_visual"
    assert first.page_count == 1
    assert first.shingle_total == 3
    assert [len(value) for value, _ in first.shingle_counts] == [16, 16, 16]
    assert len(first.exact_hashes["canonical_sha256"]) == 64
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


def test_docx_converts_to_pdf_text(tmp_path: Path) -> None:
    """Match the canonical text of a DOCX and an equivalent PDF."""
    docx = tmp_path / "one.docx"
    _docx(docx, "alpha beta gamma delta epsilon zeta 123")
    pdf = tmp_path / "one.pdf"
    _pdf(pdf, "alpha beta gamma delta epsilon zeta 123")

    office = Fingerprinter().fingerprint(docx)
    native = Fingerprinter().fingerprint(pdf)
    assert office.source_format == "docx"
    assert office.exact_hashes["canonical_sha256"] == native.exact_hashes["canonical_sha256"]


def test_image_and_blank_evidence_states(tmp_path: Path) -> None:
    """Report visual-only and insufficient-evidence outcomes explicitly."""
    path = tmp_path / "image.png"
    image = Image.new("RGB", (300, 100), "white")
    ImageDraw.Draw(image).rectangle((30, 20, 100, 80), fill="black")
    image.save(path)

    config = FingerprintConfig()
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

    bundle = Fingerprinter().fingerprint(path)
    assert bundle.source_format == "jpeg"

    transparent = tmp_path / "transparent.png"
    Image.new("RGBA", (80, 80), (0, 0, 0, 0)).save(transparent)

    bundle = Fingerprinter().fingerprint(transparent)
    assert bundle.quality.status == "insufficient_evidence"
    assert not bundle.page_hashes


def test_pdf_line_break_hyphen_joins_word(tmp_path: Path) -> None:
    """Match a word hyphenated across lines with the same word unbroken."""
    wrapped = tmp_path / "wrapped.pdf"
    document = canvas.Canvas(str(wrapped))
    lines = document.beginText(72, 720)
    lines.textLine("alpha beta inter-")
    lines.textLine("national gamma delta")
    document.drawText(lines)
    document.save()
    single = tmp_path / "single.pdf"
    _pdf(single, "alpha beta international gamma delta")

    fingerprinter = Fingerprinter()
    hashes = [fingerprinter.fingerprint(path).exact_hashes for path in (wrapped, single)]
    assert hashes[0]["canonical_sha256"] == hashes[1]["canonical_sha256"]


def test_short_native_text_does_not_trigger_ocr(tmp_path: Path) -> None:
    """Keep valid short text without using text length as an OCR trigger."""
    path = tmp_path / "short.pdf"
    _pdf(path, "Short")

    bundle = Fingerprinter().fingerprint(path)
    assert bundle.quality.pages[0].source == "native"
    assert not bundle.quality.pages[0].ocr_attempted
    assert bundle.quality.text_evidence == "exact_only"


def test_low_confidence_ocr_is_not_text_evidence() -> None:
    """Keep uncertain OCR out of the page text and report why it was rejected."""
    low = "OCR confidence was too low to use its text"
    assert _ocr_replacement(_recognized("imagined words", 5.0)) == _OCROutcome(None, 5.0, low)
    invoice = _recognized("invoice total", 30.0)
    assert _ocr_replacement(invoice) == _OCROutcome("invoice total", 30.0)
    empty = _OCROutcome(warning="OCR found no reliable text")
    assert _ocr_replacement(_recognized("", 0.0)) == empty

    header = (Box(0, 0, 40, 20),)
    body = _recognized("header alpha beta gamma", 60.0)
    assert _ocr_addition(body, header) == _OCROutcome("alpha beta gamma", 60.0)
    unconfident = "OCR did not provide confident additional text"
    body = _recognized("header alpha beta gamma", 59.9)
    assert _ocr_addition(body, header) == _OCROutcome(None, 59.9, unconfident)
    short = _recognized("header alpha", 90.0)
    assert _ocr_addition(short, header) == _OCROutcome(None, 90.0, unconfident)
    assert _ocr_addition(_recognized("header", 90.0), header) == _OCROutcome()


def test_ocr_models_are_resolved_before_fingerprinting(tmp_path: Path) -> None:
    """Use bundled or configured models and reject missing ones at construction."""
    bundled = FingerprintConfig()
    assert ocr.model_directory(bundled) == Path(ocr.__file__).with_name("tessdata")
    assert ocr.provenance(bundled)["ocr_models"].startswith("tessdata_fast@")

    with pytest.raises(FingerprintError, match="tessdata_path"):
        Fingerprinter(FingerprintConfig(ocr_language="klingon"))
    with pytest.raises(FingerprintError, match="tessdata_path lacks"):
        Fingerprinter(FingerprintConfig(tessdata_path=tmp_path))

    (tmp_path / "eng.traineddata").touch()
    custom = FingerprintConfig(tessdata_path=tmp_path)
    assert ocr.model_directory(custom) == tmp_path.resolve()
    assert ocr.provenance(custom)["ocr_models"] == "custom"


def test_latin_alphabet_codes_fall_back_to_script_latin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use script/Latin for a Latin-alphabet code only when no installed model has that code."""
    german = FingerprintConfig(ocr_language="eng+deu")
    assert ocr.provenance(german)["ocr_language"] == "eng+script/Latin"

    for name in ("eng", "deu"):
        (tmp_path / f"{name}.traineddata").touch()
    monkeypatch.setattr(ocr, "_language_models", lambda: (ocr._ModelDirectory(tmp_path, "0" * 40),))
    assert ocr.provenance(german)["ocr_language"] == "eng+deu"


def test_models_from_both_packages_load_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load script/Latin from the base package with a docfgp-languages model."""
    (tmp_path / "ara.traineddata").touch()
    monkeypatch.setattr(ocr, "_language_models", lambda: (ocr._ModelDirectory(tmp_path, "0" * 40),))

    config = FingerprintConfig(ocr_language="deu+ara")
    bundled = Path(ocr.__file__).with_name("tessdata")
    assert ocr.model_directory(config) == Path(os.path.commonpath([bundled, tmp_path]))
    assert ocr.provenance(config)["ocr_language"] == "script/Latin+ara"


def test_model_paths_that_tesseract_would_skip_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a combined model path that Tesseract would skip for its leading ~."""
    for model in ("base/script/Latin", "~languages/ara"):
        (tmp_path / f"{model}.traineddata").parent.mkdir(parents=True)
        (tmp_path / f"{model}.traineddata").touch()
    monkeypatch.setattr(ocr, "_TESSDATA", tmp_path / "base")
    monkeypatch.setattr(
        ocr, "_language_models", lambda: (ocr._ModelDirectory(tmp_path / "~languages", "0" * 40),)
    )

    with pytest.raises(FingerprintError, match="cannot load together"):
        Fingerprinter(FingerprintConfig(ocr_language="deu+ara"))


def test_language_package_models_are_discovered(tmp_path: Path) -> None:
    """Find docfgp-languages models, record their revision, and explain missing ones."""
    env = _languages_package(
        tmp_path,
        "from pathlib import Path\n"
        'TESSDATA = Path(__file__).with_name("tessdata")\n'
        'REVISION = "0123456789abcdef0123456789abcdef01234567"\n',
    )
    package = tmp_path / "docfgp_languages"
    (package / "tessdata").mkdir()
    for name in ("eng", "ara"):
        (package / "tessdata" / f"{name}.traineddata").touch()
    code = (
        "from docfgp import FingerprintConfig, ocr; "
        "config = FingerprintConfig(ocr_language='eng+ara'); "
        "print(ocr.model_directory(config)); print(ocr.provenance(config)['ocr_models'])"
    )

    directory, models = subprocess.check_output(
        [sys.executable, "-c", code], env=env, text=True
    ).split()
    assert Path(directory) == package / "tessdata"
    assert models == "tessdata_fast@0123456789abcdef0123456789abcdef01234567"

    (package / "tessdata" / "ara.traineddata").unlink()
    missing = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert "No installed OCR models for ara;" in missing.stderr

    absent = _languages_package(tmp_path / "absent", "raise ImportError\n")
    uninstalled = subprocess.run(
        [sys.executable, "-c", code], env=absent, capture_output=True, text=True, check=False
    )
    assert 'install "docfgp[languages]" for non-Latin scripts' in uninstalled.stderr


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
    with pytest.raises(FingerprintError, match="max_image_pixels"):
        Fingerprinter(FingerprintConfig(max_image_pixels=10_000)).fingerprint(image)

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

    config = FingerprintConfig()
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


def test_scanned_pdf_uses_tesseract(tmp_path: Path) -> None:
    """Recover searchable words from a PDF containing only a page image."""
    image_path = tmp_path / "scan.png"
    _scan_image(image_path)
    pdf_path = tmp_path / "scan.pdf"
    document = canvas.Canvas(str(pdf_path), pagesize=(600, 250))
    document.drawImage(str(image_path), 0, 50, width=600, height=132)
    document.save()

    bundle = Fingerprinter().fingerprint(pdf_path)
    assert bundle.quality.pages[0].source == "ocr"
    assert bundle.quality.pages[0].ocr_confidence is not None
    assert bundle.shingle_total > 0

    latin = Fingerprinter(FingerprintConfig(ocr_language="deu")).fingerprint(pdf_path)
    assert latin.quality.pages[0].source == "ocr"
    assert latin.parameters["ocr_language"] == "script/Latin"
    assert latin.parameters["ocr_engine"].startswith("tesseract-5.")


def test_rasterized_page_with_native_header_uses_ocr(tmp_path: Path) -> None:
    """Recognize substantial image text beneath a valid native header."""
    image_path = tmp_path / "body.png"
    _scan_image(image_path)
    path = tmp_path / "mixed.pdf"
    document = canvas.Canvas(str(path), pagesize=(600, 250))
    document.drawImage(str(image_path), 0, 50, width=600, height=132)
    document.drawString(20, 230, "Header")
    document.save()

    bundle = Fingerprinter().fingerprint(path)
    assert bundle.quality.pages[0].ocr_reason == "rasterized_content"
    assert bundle.quality.pages[0].source == "native_and_ocr"
    assert bundle.shingle_total > 0
