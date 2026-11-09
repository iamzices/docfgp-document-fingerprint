"""Evaluate cross-format retrieval on the office2pdf business golden corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docfgp import FingerprintConfig, Fingerprinter
from scripts._evaluation import evaluate


def _golden(
    root: Path,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]], dict[str, set[str]]]:
    """Load known Office source/export pairs from the published manifest."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = [case for case in manifest["cases"] if case["format"] in {"docx", "pptx"}]
    candidates = [(case["id"], root / case["source"]) for case in cases]
    queries = [(case["id"], root / case["expected_pdf"]) for case in cases]
    return candidates, queries, {case["id"]: {case["id"]} for case in cases}


def main(argv: list[str] | None = None) -> int:
    """Run the Office golden corpus evaluation.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Business golden corpus directory")
    parser.add_argument("--office2pdf", help="Converter executable")
    parser.add_argument("--tesseract", help="OCR executable")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    candidates, queries, truth = _golden(args.corpus)
    fingerprinter = Fingerprinter(
        FingerprintConfig(
            office2pdf_path=args.office2pdf,
            tesseract_path=args.tesseract,
            include_shingle_counts=False,
        )
    )
    report = evaluate(candidates, queries, truth, fingerprinter.fingerprint, top_k=args.top_k)
    report.update(corpus="office2pdf business golden", task="Office-to-PDF source retrieval")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
