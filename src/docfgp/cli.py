"""Expose document fingerprinting as a CLI."""

from __future__ import annotations

import argparse
import sys

from .fingerprint import Fingerprinter
from .schemas import FingerprintConfig, FingerprintError


def main(argv: list[str] | None = None) -> int:
    """Print a document fingerprint as JSON.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero on success and nonzero on processing errors.
    """
    parser = argparse.ArgumentParser(description="Fingerprint a document")
    parser.add_argument("path", help="PDF, DOCX, PPTX, or image path")
    parser.add_argument("--tessdata", help="Path to Tesseract language models")
    parser.add_argument("--ocr-language", default="eng", help="Installed Tesseract language codes")
    parser.add_argument(
        "--max-document-seconds", type=float, default=180.0, help="Processing deadline in seconds"
    )
    parser.add_argument(
        "--no-shingle-counts", action="store_true", help="Omit precise overlap counts"
    )

    args = parser.parse_args(argv)

    try:
        config = FingerprintConfig(
            tessdata_path=args.tessdata,
            ocr_language=args.ocr_language,
            max_document_seconds=args.max_document_seconds,
            include_shingle_counts=not args.no_shingle_counts,
        )
        print(Fingerprinter(config).fingerprint(args.path).to_json())
    except (FingerprintError, ValueError) as exc:
        print(f"docfgp: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
